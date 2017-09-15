from __future__ import print_function

import argparse
import yaml
from pybh.utils import fail, argparse_bool, convert_string_to_array


TYPE_STR_MAPPING = {
    "str": str,
    "int": int,
    "float": float,
    "bool": argparse_bool,
}


def run(args):
    with file(args.file, "r") as fin:
        content = yaml.load(fin)

        # keys = args.key.strip().split(".")
        keys = convert_string_to_array(args.key, sep=".")
        assert(len(keys) > 0)
        for i, key in enumerate(keys):
            if key not in content:
                if args.default is None:
                    fail("ERROR: Key #{} [{}] not found in YAML file".format(i, key))
                else:
                    content = args.default
                    break
            content = content[key]

        if args.index is not None:
            if type(content) is not list:
                # content = [x.strip() for x in content.split(",")]
                content = convert_string_to_array(content, sep=",")
            # indices = [int(x) for x in args.index.split(",")]
            indices = convert_string_to_array(args.index, sep=",", value_type=int)
            for i, index in enumerate(indices):
                if index >= len(content):
                    fail("ERROR: Index #{} ({}) out of bounds for value ({}) of type ({})".format(
                        i, index, content, type(content)))
                content = content[index]

        if args.type is not None:
            if args.type not in TYPE_STR_MAPPING:
                fail("ERROR: Type {} is unknown".format(args.type))
            else:
                try:
                    content = TYPE_STR_MAPPING[args.type](content)
                except ValueError:
                    fail("ERROR: Could not convert value ({}) of type ({}) to type {}".format(
                      content, type(content), args.type))
        # Output extracted value
        if type(content) == bool:
            content = str(content).lower()
        print(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('file', type=str, help="YAML file to read")
    parser.add_argument('key', type=str, help="Key of value to read")
    parser.add_argument('--type', type=str, help="To of value required")
    parser.add_argument('--index', type=str,
                        help="Extract element from (nested) array "
                             "(assuming the value is an array or string separated by ,)."
                             "Nested indices can be specified by separating them with ,")
    parser.add_argument('--default', type=str, help="Default value if key is not in YAML file")

    args = parser.parse_args()

    run(args)
