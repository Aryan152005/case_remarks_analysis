import json

FIELD_VALUE_SCHEMA = {
    "event_type": [
        'VIOLENT CRIME', 'THEFT & BURGLARY', 'PUBLIC DISTURBANCE', 'FIRE & HAZARDS',
        'RESCUE OPERATIONS', 'MEDICAL EMERGENCIES', 'TRAFFIC INCIDENTS',
        'PUBLIC NUISANCE', 'SOCIAL ISSUES', 'MISSING PERSONS',
        'NATURAL INCIDENTS', 'OTHERS'
    ],
    "event_sub_type": {
        'VIOLENT CRIME': [
            'ASSAULT', 'KIDNAPPING', 'BOMB BLAST', 'CHILD ABUSE', 'CRIME AGAINST WOMEN',
            'ROBBERY', 'DEAD BODY FOUND', 'SUICIDE ATTEMPT', 'MURDER', 'THREAT',
            'DOMESTIC VIOLENCE', 'VERBAL ABUSE'
        ],
        'THEFT & BURGLARY': [
            'THEFT', 'ATTEMPT OF THEFT', 'HOUSE BREAKING ATTEMPTS', 'VEHICLE THEFT'
        ],
        'PUBLIC DISTURBANCE': [
            'SCUFFLE AMONG STUDENTS', 'DRUNKEN ATROCITIES', 'VERBAL ABUSE', 'GAMBLING',
            'NUDITY IN PUBLIC', 'STRIKE', 'GENERAL NUISANCE'
        ],
        'FIRE & HAZARDS': [
            'FIRE', 'ELECTRICAL FIRE', 'BUILDING FIRE', 'LANDSCAPE FIRE', 'VEHICLE FIRE',
            'GAS LEAKAGE', 'HAZARDOUS CONDITION INCIDENTS'
        ],
        'RESCUE OPERATIONS': [
            'WATER RESCUE', 'WELL RESCUE', 'SEARCH AND RESCUE', 'ROAD CRASH RESCUE'
        ],
        'MEDICAL EMERGENCIES': [
            'BREATHING DIFFICULTIES', 'PERSON COLLAPSED', 'AMBULANCE SERVICE',
            'INTER HOSPITAL TRANSFER', 'BLEEDING', 'HEART ATTACK', 'FIRE INJURY'
        ],
        'TRAFFIC INCIDENTS': [
            'HIT & RUN INCIDENTS', 'RASH DRIVING', 'TRAFFIC BLOCK', 'OBSTRUCTIVE PARKING',
            'VEHICLE BREAK DOWN', 'ACCIDENT', 'RUN OVER INCIDENTS'
        ],
        'PUBLIC NUISANCE': [
            'ILLEGAL MINING', 'ILLEGAL CONSTRUCTIONS', 'GENERAL NUISANCE',
            'WASTAGE DUMPING ISSUE', 'AIR POLLUTION', 'NOISE POLLUTION',
            'TRESPASSING TO PROPERTY'
        ],
        'SOCIAL ISSUES': [
            'FAMILY ISSUES', 'LABOUR/WAGES ISSUES', 'DISPUTES BETWEEN NEIGHBOURS',
            'LAND BOUNDARY ISSUES', 'ISSUES RELATED TO SENIOR CITIZENS',
            'MIGRANT LABOURERS ISSUES'
        ],
        'MISSING PERSONS': [
            'MISSING', 'SUSPICIOUSLY FOUND PERSONS OR VEHICLES', 'CHILD LINES'
        ],
        'NATURAL INCIDENTS': [
            'RAINY SEASON INCIDENTS', 'NIZHAL PANIC CALL', 'FLOOD', 'DISASTER',
            'EARTHQUAKE', 'LANDSLIDE', 'STRUCTURE COLLAPSE'
        ],
        'OTHERS': [
            'OTHERS', 'SALE OF CONTRABANDS', 'ASSISTANCE FOR HOSPITALIZATION OF CHALLENGED PERSONS',
            'CYBER CRIME', 'RAILWAY', 'ABANDONED VEHICLES'
        ]
    },
    "state_of_victim": [
        'Distressed', 'Stable', 'Injured', 'Critical', 'Unconscious', 'Deceased', 'Drunken', 'Not specified'
    ],
    "victim_gender": [
        'male', 'female', 'not specified'
    ],
    "specified_matter": "text_allow_not_specified",
    "date_reference": "text_allow_not_specified",
    "frequency": "text_allow_not_specified",
    "repeat_incident": "text_allow_not_specified",
    "identification": "text_allow_not_specified",
    "injury_type": "text_allow_not_specified",
    "victim_age": "text_allow_not_specified",
    "victim_relation": "text_allow_not_specified",
    "incident_location": "text_allow_not_specified",
    "area": "text_allow_not_specified",
    "suspect_description": "text_allow_not_specified",
    "object_involved": "text_allow_not_specified",
    "used_weapons": "text_allow_not_specified",
    "offender_relation": "text_allow_not_specified",
    "mode_of_threat": "text_allow_not_specified",
    "need_ambulance": "text_allow_not_specified",
    "children_involved": "text_allow_not_specified"
}

# Ensure all 'event_sub_type' values are flattened and unique for validation purposes later if needed
ALL_EVENT_SUB_TYPES = sorted(list(set(
    item for sublist in FIELD_VALUE_SCHEMA["event_sub_type"].values() for item in sublist
)))

# List of fields that are expected to be present in the output
# This also defines the order for JSON output and CSV columns
ALL_CLASSIFICATION_FIELDS = [
    "event_type", "event_sub_type", "state_of_victim", "victim_gender",
    "specified_matter", "date_reference", "frequency", "repeat_incident",
    "identification", "injury_type", "victim_age", "victim_relation",
    "incident_location", "area", "suspect_description", "object_involved",
    "used_weapons", "offender_relation", "mode_of_threat", "need_ambulance",
    "children_involved"
]

if __name__ == '__main__':
    print("Schema Definitions:")
    print(json.dumps(FIELD_VALUE_SCHEMA, indent=4))
    print("\nAll Event Sub Types (Flattened for Prompt):", ALL_EVENT_SUB_TYPES)
    print("\nAll Classification Fields (Ordered):", ALL_CLASSIFICATION_FIELDS)

    # Example of how to derive event_type from event_sub_type programmatically
    def derive_event_type(sub_type: str) -> str:
        sub_type_upper = sub_type.upper()
        for event_type_key, sub_types_list in FIELD_VALUE_SCHEMA['event_sub_type'].items():
            if sub_type_upper in [st.upper() for st in sub_types_list]:
                return event_type_key
        return 'OTHERS' # Default if sub_type not found

    print("\nDerived event_type for 'THEFT':", derive_event_type('THEFT'))
    print("Derived event_type for 'ACCIDENT':", derive_event_type('ACCIDENT'))
    print("Derived event_type for 'MYSTERY_SUBTYPE':", derive_event_type('MYSTERY_SUBTYPE'))
    print("Derived event_type for 'OTHERS':", derive_event_type('OTHERS'))
    print("Derived event_type for 'general nuisance':", derive_event_type('general nuisance'))