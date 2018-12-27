import argparse
import pathlib


def to_csv(path):
    files = [p.as_posix() for p in path.glob('*.split')]
    if args.output:
        files = [f.replace('.split', '.split.input') for f in files]
    return ','.join(files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('--output', action='store_true')
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    print(to_csv(data_dir))
