import argparse

def parse_args():
	parser = argparse.ArgumentParser(description='Arguments')
	parser.add_argument('--anime_planning', action='store_true', help='show animation when planning')
	return parser.parse_args()

args = parse_args()

