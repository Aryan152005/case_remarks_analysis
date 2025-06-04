# schema.py

FIELD_VALUE_SCHEMA = {
    "event_type": [
        'VIOLENT CRIME', 'THEFT & BURGLARY', 'PUBLIC DISTURBANCE', 'FIRE & HAZARDS',
        'RESCUE OPERATIONS', 'MEDICAL EMERGENCIES', 'TRAFFIC INCIDENTS',
        'PUBLIC NUISANCE', 'SOCIAL ISSUES', 'MISSING PERSONS',
        'NATURAL INCIDENTS', 'OTHERS'
    ],
    "event_sub_type": {
        'VIOLENT CRIME': ['ASSAULT', 'KIDNAPPING', 'BOMB BLAST', 'CHILD ABUSE', 'CRIME AGAINST WOMEN', 'ROBBERY', 'DEAD BODY FOUND', 'SUICIDE ATTEMPT', 'MURDER', 'THREAT', 'DOMESTIC VIOLENCE', 'VERBAL ABUSE'],
        'THEFT & BURGLARY': ['THEFT', 'ATTEMPT OF THEFT', 'HOUSE BREAKING ATTEMPTS', 'VEHICLE THEFT'],
        'PUBLIC DISTURBANCE': ['SCUFFLE AMONG STUDENTS', 'DRUNKEN ATROCITIES', 'VERBAL ABUSE', 'GAMBLING', 'NUDITY IN PUBLIC', 'STRIKE', 'GENERAL NUISSANCE'],
        'FIRE & HAZARDS': ['FIRE', 'ELECTRICAL FIRE', 'BUILDING FIRE', 'LANDSCAPE FIRE', 'VEHICLE FIRE', 'GAS LEAKAGE', 'HAZARDOUS CONDITION INCIDENTS'],
        'RESCUE OPERATIONS': ['WATER RESCUE', 'WELL RESCUE', 'SEARCH AND RESCUE', 'ROAD CRASH RESCUE'],
        'MEDICAL EMERGENCIES': ['BREATHING DIFFICULTIES', 'PERSON COLLAPSED', 'AMBULANCE SERVICE', 'INTER HOSPITAL TRANSFER', 'BLEEDING', 'HEART ATTACK', 'FIRE INJURY'],
        'TRAFFIC INCIDENTS': ['HIT & RUN INCIDENTS', 'RASH DRIVING', 'TRAFFIC BLOCK', 'OBSTRUCTIVE PARKING', 'VEHICLE BREAK DOWN', 'ACCIDENT', 'RUN OVER INCIDENTS'],
        'PUBLIC NUISANCE': ['ILLEGAL MINING', 'ILLEGAL CONSTRUCTIONS', 'GENERAL NUISSANCE', 'WASTAGE DUMPING ISSUE', 'AIR POLLUTION', 'NOISE POLUTION', 'TRESSPASSING TO PROPERTY', 'PUBLIC NUISANCE'],
        'SOCIAL ISSUES': ['FAMILY ISSUES', 'LABOUR/WAGES ISSUES', 'DISPUTES BETWEEN NEIGHBOURS', 'LAND BOUNDARY ISSUES', 'ISSUES RELATED TO SENIOR CITIZENS', 'MIGRANT LABOURERS ISSUES'],
        'MISSING PERSONS': ['MISSING', 'SUSPECIOUSLY FOUND PERSONS OR VEHICLES', 'CHILD LINES'],
        'NATURAL INCIDENTS': ['RAINY SEASON INCIDENTS', 'NIZHAL PANIC CALL', 'FLOOD', 'DISASTER', 'EARTHQUAKE', 'LANDSLIDE', 'STRUCTURE COLLAPSE'],
        'OTHERS': ['OTHERS', 'SALE OF CONTRABANDS', 'ASSISTANCE FOR HOSPITALIZATION OF CHALLENGED PERSONS', 'CYBER CRIME', 'RAILWAY', 'ABANDONED VEHICLES']
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