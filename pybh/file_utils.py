import os
import re


def get_next_filename(filename_template, start_file_num=0):
    file_num = start_file_num
    while True:
        filename = filename_template.format(file_num)
        if not os.path.isfile(filename):
            break
        file_num += 1
    return filename, file_num


def get_matching_filenames(filename_pattern, path=".", return_match_objects=False):
    re_pattern = re.compile(filename_pattern)
    matching_filenames = []
    if return_match_objects:
        match_objects = []
    for filename in os.listdir(path):
        match = re_pattern.match(filename)
        if match:
            matching_filenames.append(filename)
            if return_match_objects:
                match_objects.append(match)
    if return_match_objects:
        return zip(matching_filenames, match_objects)
    else:
        return matching_filenames
