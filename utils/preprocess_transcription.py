import os
import re
from collections import defaultdict

def remove_ads(transcript: str) -> str:
    # Split the transcript into lines
    lines = transcript.split('\n')

    # List to hold lines that are part of the main content, including speaker identifiers
    content_lines = []

    # Adjusted regular expression to match speaker identifiers with hours, minutes, and seconds
    speaker_regex = r'^\d+ \((\d+h )?(\d+m )?\d+s\):'

    # Loop through the lines with an index so we can look ahead
    i = 0
    while i < len(lines):
        # Check if the line matches the pattern for a speaker
        if re.match(speaker_regex, lines[i]):
            # Include this line (speaker identifier)
            content_lines.append(lines[i])
            
            # Check and include the next line if it exists
            if i + 1 < len(lines):
                content_lines.append(lines[i + 1])  # Include the next line (content)
                content_lines.append("")  # Add an empty line for separation
                
            i += 2  # Move past the next line since we've included it
        else:
            i += 1  # Increment to check the next line if this isn't a speaker

    # Join the content lines back into a single string, ensuring separation
    cleaned_transcript = '\n'.join(content_lines)
    return cleaned_transcript

def identify_host(transcript: str) -> (str, bool):
    # Split the transcript into lines
    lines = transcript.split('\n')

    # Dictionaries to hold counts
    welcome_counts = defaultdict(int)
    question_counts = defaultdict(int)

    # Updated regular expression to match speaker identifiers with any timestamp
    speaker_regex = r'^(\d+) \((\d+h )?\d+m \d+s\):'

    current_speaker = None
    for line in lines:
        # Check if the line contains a speaker identifier
        speaker_match = re.match(speaker_regex, line)
        if speaker_match:
            current_speaker = speaker_match.group(1)
        else:
            # If the line doesn't contain a speaker identifier, it's part of the current speaker's text
            if current_speaker is not None:
                # Count 'welcome to' and 'welcome back'
                if 'welcome to' in line.lower() or 'welcome back' in line.lower():
                    welcome_counts[current_speaker] += 1

                # Count question marks
                question_counts[current_speaker] += line.count('?')

    # Identify the host and the speaker with the most questions
    # If multiple speakers have the same welcome count, this method still picks one, 
    # which might need manual verification for accuracy
    welcome_speaker = max(welcome_counts, key=welcome_counts.get) if welcome_counts else None
    most_questions_speaker = max(question_counts, key=question_counts.get) if question_counts else None

    # check if the welcome_speaker and most_questions_speaker is the same if they are then welcome_questions_match = True else False
    if welcome_speaker == most_questions_speaker:
        host_speaker = welcome_speaker
        welcome_questions_match = True
    elif welcome_speaker is None:
        host_speaker = most_questions_speaker
        welcome_questions_match = False
    else:
        host_speaker = welcome_speaker
        welcome_questions_match = False

    return (host_speaker, welcome_questions_match)

def insert_marker_before_host(transcript: str, host_speaker: int) -> str:
    # Split the transcript into lines
    lines = transcript.split('\n')

    # Regular expression to match any speaker identifiers with timestamps
    speaker_regex = r'^(\d+) \((\d+h )?\d+m \d+s\):'

    # List to hold the modified lines
    modified_lines = []

    for line in lines:
        speaker_match = re.match(speaker_regex, line)
        if speaker_match:
            current_speaker = speaker_match.group(1)
            # If the current speaker is the host, insert the marker before adding the line
            if current_speaker == host_speaker:
                modified_lines.append("###")  # Insert marker
            modified_lines.append(line)
        else:
            modified_lines.append(line)

    # Join the modified lines back into a single string
    modified_transcript = '\n'.join(modified_lines)
    return modified_transcript

