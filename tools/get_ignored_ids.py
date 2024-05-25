import sys
from run_dcompare2 import get_ignored_ids

if __name__ == "__main__":
    path = sys.argv[1]
    # print("Analyzing", path)
    ignore = get_ignored_ids(path)
    # print(f"Ignoring {len(ignore)}")
    for k in ignore:
        print(k)