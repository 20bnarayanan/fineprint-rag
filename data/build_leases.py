"""Generate synthetic apartment leases with planted "traps" and ground truth.

Everything here is fictional. The leases are NOT legal documents, and nothing in
this project is legal advice. Each trap is a clause (or a pair of conflicting
clauses) that a careful tenant would want to notice. The builder records the
exact evidence sentence(s) for every trap, so evaluation can check mechanically
whether a plan found it.

Run:  python data/build_leases.py
Writes data/leases/<lease_id>.txt and data/traps.json
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent

# --------------------------------------------------------------------------
# Trap metadata: id -> (category, severity, one-line summary)
# --------------------------------------------------------------------------
TRAP_META: dict[str, tuple[str, str, str]] = {
    "auto_renewal": ("renewal", "high", "Lease auto-renews for a full year unless notice is given far in advance."),
    "rent_escalation": ("rent_increase", "medium", "Landlord can raise rent on renewal by up to a fixed percentage."),
    "late_fee_conflict": ("late_fees", "medium", "Two sections disagree about when rent is late and what the fee is."),
    "deposit_conflict": ("security_deposit", "medium", "Two sections disagree about how fast the deposit is returned."),
    "nonrefundable_fee": ("security_deposit", "medium", "Part of the deposit is declared non-refundable."),
    "utility_allocation": ("utilities", "medium", "Landlord allocates shared utility costs at its sole discretion."),
    "repair_cost_shift": ("maintenance", "high", "Tenant pays the first part of every repair regardless of cause."),
    "entry_no_notice": ("landlord_entry", "high", "Landlord may enter at any time without notice."),
    "guest_limit": ("guests", "low", "Guests are limited to a few nights per month."),
    "pet_fees": ("pets", "medium", "Recurring pet rent plus a non-refundable per-animal fee."),
    "sublet_penalty": ("subletting", "medium", "Subletting, including short-term rentals, triggers termination and a penalty."),
    "early_termination_accel": ("early_termination", "high", "Early termination accelerates all remaining rent with no duty to re-rent."),
    "attorney_fees_oneway": ("legal_fees", "medium", "Tenant pays landlord's attorney fees regardless of who wins."),
    "arbitration_waiver": ("dispute_resolution", "high", "Jury trial waiver and binding arbitration with tenant paying arbitrator fees."),
}

# --------------------------------------------------------------------------
# Section library. Each slot: benign text (or None), optional trap texts keyed
# by trap id, and evidence_for (trap ids for which the *benign* sentence is
# also evidence, used for conflict traps).
# --------------------------------------------------------------------------
Slot = dict
SECTIONS: list[tuple[str, list[Slot]]] = [
    ("PARTIES AND PREMISES", [
        {"benign": "This Lease is made between {landlord} (\"Landlord\") and {tenant} (\"Tenant\") for the residential unit at {address} (the \"Premises\")."},
        {"benign": "The Premises are leased for use as a private residence only."},
    ]),
    ("TERM AND RENEWAL", [
        {"benign": "The term of this Lease begins on {start} and ends on {end}."},
        {"benign": "At the end of the term, the tenancy converts to a month-to-month tenancy unless either party gives the other at least 30 days' written notice.",
         "traps": {"auto_renewal": "This Lease automatically renews for a new 12-month term unless Tenant delivers written notice of non-renewal by certified mail at least 90 days before the end of the then-current term."}},
    ]),
    ("RENT AND PAYMENT", [
        {"benign": "Tenant shall pay monthly rent of ${rent}, due on the first day of each month."},
        {"benign": "Rent received after the 5th day of the month is subject to a late fee of 5% of the monthly rent.",
         "evidence_for": ["late_fee_conflict"]},
        {"benign": "Payment may be made by electronic transfer or by check delivered to the management office.",
         "traps": {}},
        {"benign": None,
         "traps": {"rent_escalation": "Upon each renewal, Landlord may increase the monthly rent by up to 8% with 30 days' written notice."}},
    ]),
    ("SECURITY DEPOSIT", [
        {"benign": "Tenant shall pay a security deposit of ${deposit} before taking possession."},
        {"benign": "Landlord shall return the deposit, less lawful deductions for unpaid rent and damage beyond normal wear and tear, within 21 days after Tenant vacates and returns all keys.",
         "evidence_for": ["deposit_conflict"]},
        {"benign": None,
         "traps": {"nonrefundable_fee": "Of the deposit, $400 is a non-refundable cleaning fee that Landlord keeps regardless of the condition of the Premises."}},
    ]),
    ("UTILITIES AND SERVICES", [
        {"benign": "Tenant shall pay for electricity and internet service used at the Premises and shall place those accounts in Tenant's name."},
        {"benign": "Landlord shall pay for water, sewer, and trash service.",
         "traps": {"utility_allocation": "Tenant shall pay a share of common-area water, sewer, and trash charges, as allocated by Landlord in its sole discretion."}},
    ]),
    ("MAINTENANCE AND REPAIRS", [
        {"benign": "Tenant shall keep the Premises clean and shall promptly report any needed repairs to Landlord in writing."},
        {"benign": "Landlord shall make repairs to the plumbing, heating, and electrical systems at Landlord's expense, except for damage caused by Tenant.",
         "traps": {"repair_cost_shift": "Tenant is responsible for the first $300 of the cost of any repair to the Premises, regardless of the cause of the damage."}},
    ]),
    ("LANDLORD ENTRY", [
        {"benign": "Landlord may enter the Premises to inspect, make repairs, or show the unit to prospective tenants after giving Tenant at least 24 hours' notice.",
         "traps": {"entry_no_notice": "Landlord may enter the Premises at any time and without prior notice for inspections, repairs, or showings."}},
        {"benign": "In an emergency, Landlord may enter without notice."},
    ]),
    ("OCCUPANCY AND GUESTS", [
        {"benign": "Only Tenant and those persons listed on the application may reside in the Premises."},
        {"benign": "Guests may stay for up to 14 consecutive days without Landlord approval.",
         "traps": {"guest_limit": "Any guest who stays more than 5 nights in a calendar month is considered an unauthorized occupant and requires Landlord's written approval."}},
    ]),
    ("PETS", [
        {"benign": "No animals may be kept in the Premises without Landlord's prior written consent.",
         "evidence_for": []},
        {"benign": None,
         "traps": {"pet_fees": "If Landlord consents to a pet, Tenant shall pay monthly pet rent of $50 per animal and a non-refundable pet fee of $500 per animal."}},
    ]),
    ("SUBLETTING AND ASSIGNMENT", [
        {"benign": "Tenant shall not sublet the Premises or assign this Lease without Landlord's prior written consent, which shall not be unreasonably withheld.",
         "traps": {"sublet_penalty": "Tenant shall not sublet the Premises or any part of it, including through short-term rental platforms, and any violation is grounds for immediate termination and a penalty of $1,000."}},
    ]),
    ("INSURANCE", [
        {"benign": "Tenant shall maintain renter's insurance with at least $100,000 of liability coverage throughout the term."},
        {"benign": "Tenant shall deliver proof of renter's insurance to Landlord within 10 days after the start of the term."},
    ]),
    ("MOVE-IN INSPECTION", [
        {"benign": "Landlord and Tenant shall inspect the Premises together at move-in."},
        {"benign": "Tenant must return the signed move-in condition checklist to Landlord within 7 days after move-in; otherwise the Premises are deemed to be in good condition."},
    ]),
    ("EARLY TERMINATION", [
        {"benign": "Tenant may terminate this Lease before the end of the term by giving 60 days' written notice and paying a termination fee equal to two months' rent.",
         "traps": {"early_termination_accel": "If Tenant terminates this Lease early for any reason, Tenant shall owe all rent remaining through the end of the term, and Landlord has no duty to re-rent the Premises."}},
    ]),
    ("DEFAULT AND LATE CHARGES", [
        {"benign": "If Tenant fails to pay rent or violates this Lease, Landlord may give written notice of default and a reasonable period to cure."},
        {"benign": None,
         "traps": {"late_fee_conflict": "Rent not received by the 2nd day of the month is delinquent and subject to a late charge of $75 plus $15 for each additional day."}},
    ]),
    ("DISPUTE RESOLUTION AND FEES", [
        {"benign": "The parties shall first try to resolve any dispute through good-faith discussion.",
         "traps": {"arbitration_waiver": "Tenant waives the right to a jury trial and agrees that all disputes shall be resolved by binding arbitration, with Tenant paying the arbitrator's fees."}},
        {"benign": None,
         "traps": {"attorney_fees_oneway": "Tenant shall pay Landlord's reasonable attorney's fees in any dispute arising under this Lease, regardless of which party prevails."}},
    ]),
    ("MOVE-OUT", [
        {"benign": "At move-out, Tenant shall return the Premises in the same condition as at move-in, except for normal wear and tear, and shall return all keys."},
        {"benign": "Tenant shall provide a forwarding address in writing."},
    ]),
    ("GENERAL PROVISIONS", [
        {"benign": "This Lease is the entire agreement between the parties and may be changed only in a writing signed by both."},
        {"benign": None,
         "traps": {"deposit_conflict": "Notwithstanding any other provision of this Lease, Landlord shall return any security deposit within 60 days after Tenant vacates."}},
        {"benign": "Notices must be in writing and delivered to the address listed above."},
    ]),
]

# Ground-truth deadlines present in every lease (not traps, but expected in plans).
DEADLINES = [
    {"label": "Return signed move-in condition checklist", "anchor": "move_in", "offset_days": 7,
     "evidence": "Tenant must return the signed move-in condition checklist to Landlord within 7 days after move-in; otherwise the Premises are deemed to be in good condition."},
    {"label": "Deliver proof of renter's insurance", "anchor": "lease_start", "offset_days": 10,
     "evidence": "Tenant shall deliver proof of renter's insurance to Landlord within 10 days after the start of the term."},
]

LEASES = [
    {"id": "lease_a", "landlord": "Maple Court Holdings LLC", "tenant": "Jordan Rivera",
     "address": "214 Maple Court, Unit 3B, Springfield", "rent": "2,450", "deposit": "2,450",
     "start": "January 1, 2027", "end": "December 31, 2027",
     "traps": ["auto_renewal", "repair_cost_shift", "entry_no_notice", "late_fee_conflict"]},
    {"id": "lease_b", "landlord": "Harbor View Properties Inc.", "tenant": "Priya Desai",
     "address": "88 Harbor View Drive, Apt 12, Riverton", "rent": "3,100", "deposit": "3,100",
     "start": "February 1, 2027", "end": "January 31, 2028",
     "traps": ["early_termination_accel", "nonrefundable_fee", "attorney_fees_oneway", "deposit_conflict"]},
    {"id": "lease_c", "landlord": "Cedar Lofts Management", "tenant": "Marcus Lee",
     "address": "5 Cedar Street, Loft 4, Lakeside", "rent": "1,900", "deposit": "1,900",
     "start": "March 1, 2027", "end": "February 29, 2028",
     "traps": ["sublet_penalty", "utility_allocation", "arbitration_waiver", "rent_escalation"]},
    {"id": "lease_d", "landlord": "Willow Row Partners", "tenant": "Elena Petrov",
     "address": "40 Willow Row, Unit 7, Brookfield", "rent": "2,750", "deposit": "2,750",
     "start": "April 1, 2027", "end": "March 31, 2028",
     "traps": ["pet_fees", "guest_limit", "auto_renewal", "deposit_conflict"]},
    {"id": "lease_e_clean", "landlord": "Juniper House LLC", "tenant": "Sam Okafor",
     "address": "9 Juniper Lane, Unit 2, Fairview", "rent": "2,200", "deposit": "2,200",
     "start": "May 1, 2027", "end": "April 30, 2028",
     "traps": []},
]


def build_lease(cfg: dict) -> tuple[str, dict]:
    chosen = set(cfg["traps"])
    params = {k: cfg[k] for k in ("landlord", "tenant", "address", "rent", "deposit", "start", "end")}
    evidence: dict[str, list[str]] = {t: [] for t in chosen}
    lines = ["RESIDENTIAL LEASE AGREEMENT (SYNTHETIC SAMPLE, NOT A REAL DOCUMENT)", ""]
    for n, (title, slots) in enumerate(SECTIONS, start=1):
        sentences = []
        for slot in slots:
            trap_here = [t for t in slot.get("traps", {}) if t in chosen]
            if trap_here:
                tid = trap_here[0]
                text = slot["traps"][tid].format(**params)
                evidence[tid].append(text)
            elif slot.get("benign"):
                text = slot["benign"].format(**params)
                for tid in slot.get("evidence_for", []):
                    if tid in chosen:
                        evidence[tid].append(text)
            else:
                continue
            sentences.append(text)
        lines.append(f"SECTION {n}. {title}. " + " ".join(sentences))
        lines.append("")
    text = "\n".join(lines).rstrip() + "\n"
    truth = {
        "lease_id": cfg["id"],
        "traps": [
            {"id": t, "category": TRAP_META[t][0], "severity": TRAP_META[t][1],
             "summary": TRAP_META[t][2], "evidence": evidence[t]}
            for t in cfg["traps"]
        ],
        "deadlines": DEADLINES,
        "lease_start": cfg["start"],
        "lease_end": cfg["end"],
    }
    return text, truth


def main() -> None:
    (HERE / "leases").mkdir(exist_ok=True)
    all_truth = {}
    for cfg in LEASES:
        text, truth = build_lease(cfg)
        (HERE / "leases" / f"{cfg['id']}.txt").write_text(text)
        all_truth[cfg["id"]] = truth
        print(f"wrote {cfg['id']}: {len(text.split())} words, {len(truth['traps'])} traps")
    (HERE / "traps.json").write_text(json.dumps(all_truth, indent=2) + "\n")


if __name__ == "__main__":
    main()
