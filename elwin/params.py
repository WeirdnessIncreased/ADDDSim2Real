import argparse

def parse_args():
	parser = argparse.ArgumentParser(description='Arguments')
	parser.add_argument('--anime_plan', action='store_true', help='show animation when planning')
	parser.add_argument('--anime_curv', action='store_true', help='show curvature of a path')
	parser.add_argument('--anime_dyob', action='store_true', help='show dynamic obstacles')
	parser.add_argument('--anime_run', action='store_true', help='show real time running animation')
	parser.add_argument('--no_graphics', action='store_true', help='don\'t show simulator graphics')
	return parser.parse_args()

args = parse_args()

