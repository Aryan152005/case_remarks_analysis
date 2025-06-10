import json

# Example 1: Demonstrating a clear classification
# This example is illustrative. You should replace it with real examples from your ground truth.
EXAMPLE_1_JSON = {
    "file_name": "audio_sample_1",
    "event_info_text": "Madam, there's been a bad accident near the marketplace. A car hit a motorcycle and the person is badly injured. They need an ambulance immediately. The car drove off.",
    "event_type": "TRAFFIC INCIDENTS", # This will be derived later by validation, but it's what we expect.
    "event_sub_type": "ACCIDENT",
    "state_of_victim": "Injured",
    "victim_gender": "not specified",
    "specified_matter": "Car hit motorcycle, person injured, needs ambulance. Car fled.",
    "date_reference": "not specified",
    "frequency": "not specified",
    "repeat_incident": "not specified",
    "identification": "not specified",
    "injury_type": "not specified", # Can be more specific if text says "bleeding", "fracture" etc.
    "victim_age": "not specified",
    "victim_relation": "not specified",
    "incident_location": "marketplace",
    "area": "not specified",
    "suspect_description": "car drove off",
    "object_involved": "car, motorcycle",
    "used_weapons": "not specified", # Or "none" if implied
    "offender_relation": "not specified",
    "mode_of_threat": "HIT & RUN INCIDENTS", # Based on previous example of "hit and run"
    "need_ambulance": "yes",
    "children_involved": "not specified"
}

# Example 2: Demonstrating the 'OTHERS' event_sub_type within 'OTHERS' event_type
EXAMPLE_2_JSON = {
    "file_name": "audio_sample_2",
    "event_info_text": "Hello, I called because my neighbor keeps throwing trash onto my property. It's becoming a regular issue.",
    "event_type": "PUBLIC NUISANCE", # This will be derived.
    "event_sub_type": "WASTAGE DUMPING ISSUE",
    "state_of_victim": "Distressed",
    "victim_gender": "not specified",
    "specified_matter": "Neighbor repeatedly dumping trash on caller's property.",
    "date_reference": "not specified",
    "frequency": "not specified", # Assuming it's not specific to one time
    "repeat_incident": "yes",
    "identification": "not specified",
    "injury_type": "not specified",
    "victim_age": "not specified",
    "victim_relation": "neighbor",
    "incident_location": "caller's property",
    "area": "not specified",
    "suspect_description": "neighbor",
    "object_involved": "trash",
    "used_weapons": "not specified",
    "offender_relation": "neighbor",
    "mode_of_threat": "not specified",
    "need_ambulance": "no",
    "children_involved": "not specified"
}

# Example 3: Incident that falls under 'OTHERS' -> 'SALE OF CONTRABANDS'
EXAMPLE_3_JSON = {
    "file_name": "audio_sample_3",
    "event_info_text": "There's a group of people selling illegal drugs openly in the park. It's happening right now.",
    "event_type": "OTHERS", # Derived from sub_type
    "event_sub_type": "SALE OF CONTRABANDS",
    "state_of_victim": "not specified",
    "victim_gender": "not specified",
    "specified_matter": "Group selling illegal drugs in the park.",
    "date_reference": "currently",
    "frequency": "not specified",
    "repeat_incident": "not specified",
    "identification": "not specified",
    "injury_type": "not specified",
    "victim_age": "not specified",
    "victim_relation": "not specified",
    "incident_location": "park",
    "area": "not specified",
    "suspect_description": "group of people",
    "object_involved": "illegal drugs",
    "used_weapons": "not specified",
    "offender_relation": "not specified",
    "mode_of_threat": "not specified",
    "need_ambulance": "no",
    "children_involved": "not specified"
}


FEW_SHOT_EXAMPLES = [
    EXAMPLE_1_JSON,
    EXAMPLE_2_JSON,
    EXAMPLE_3_JSON, # Include this to show specific OTHERS subtype handling
]

def get_few_shot_examples_str():
    """Formats the FEW_SHOT_EXAMPLES into a string for the prompt."""
    examples_str_list = []
    for i, ex_json in enumerate(FEW_SHOT_EXAMPLES):
        # The `event_info_text` is the input, the rest is the expected output.
        example_text = ex_json.pop("event_info_text")
        # LLM does not output 'file_name', so remove it from the example JSON for LLM's expected output
        file_name_for_display = ex_json.pop("file_name")

        examples_str_list.append(f"**Example {i+1} (File: {file_name_for_display}):**")
        examples_str_list.append(f"Input Transcript:\n```\n{example_text}\n```")
        examples_str_list.append("Expected Output JSON (one field per line):")
        
        # Manually format for the prompt's expected output style (field_name: value)
        for field, value in ex_json.items():
            examples_str_list.append(f"{field}: {value}")
        examples_str_list.append("") # Add a blank line for separation

        # Re-add popped fields if you need the original FEW_SHOT_EXAMPLES list to remain intact for other uses
        ex_json["event_info_text"] = example_text
        ex_json["file_name"] = file_name_for_display

    return "\n".join(examples_str_list)

if __name__ == '__main__':
    print(get_few_shot_examples_str())