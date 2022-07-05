import numpy as np
import random, sys, math, os.path
from matplotlib.pyplot import imread
from matplotlib import pyplot as ppl
from matplotlib import cm

MIN_NUM_VERT = 20 # Minimum number of vertex in the graph
MAX_NUM_VERT = 1500 # Maximum number of vertex in the graph
STEP_DIS = 20 # Maximum distance between two vertex
SEED = None # For random numbers

def find_Nearest_Point(points, point):
    best = (sys.maxsize, sys.maxsize, sys.maxsize)
    for p in points:
        if p == point:
            continue
        dist = math.sqrt( ( p[0] - point[0] ) ** 2 + ( p[1] - point[1]) ** 2 )
        if dist < best[2]:
            best = (p[0], p[1], dist)
    return (best[0], best[1])

def connect_Points(a, b, img):
    newPoints = []
    newPoints.append([ b[0], b[1] ])
    step = [ (a[0] - b[0]) / float(STEP_DIS), (a[1] - b[1]) / float(STEP_DIS) ]

    # Set small steps to check for walls
    pointsNeeded = int(math.floor(max(math.fabs(step[0]), math.fabs(step[1]))))

    if ( math.fabs(step[0]) > math.fabs(step[1]) ):
        if( step[0] >= 0 ):
            step = [ 1, step[1] / math.fabs(step[0]) ]
        else:
            step = [ -1, step[1] / math.fabs(step[0]) ]

    else:
        if( step[1] >= 0 ):
            step = [ step[0] / math.fabs(step[1]), 1 ]
        else:
            step = [ step[0]/math.fabs(step[1]), -1 ]

    blocked = False
    for i in range( pointsNeeded + 1 ): # Creates points between graph and solitary point
        for j in range(STEP_DIS): # Check if there are walls between points
            coordX = round(newPoints[i][0] + step[0] * j)
            coordY = round(newPoints[i][1] + step[1] * j)

            if( coordX == a[0] and coordY == a[1] ):
                break
            if( coordY >= len(img) or coordX >= len(img[0]) ):
                break
            if( img[int(coordY)][int(coordX)][0] * 255 < 255 ):
                blocked = True
            if blocked:
                break

        if blocked:
            break
        if not (coordX == a[0] and coordY == a[1]):
            newPoints.append([ newPoints[i][0] + (step[0] * STEP_DIS), newPoints[i][1] + (step[1] * STEP_DIS) ])

    if not blocked:
        newPoints.append([ a[0], a[1] ])

    return newPoints

def searchPath(graph, point, path):
    for i in graph:
        if point == i[0]:
            p = i

        if p[0] == graph[-1][0]:
            return path

    for link in p[1]:
        path.append(link)
        finalPath = searchPath(graph, link, path)

        if finalPath != None:
            return finalPath
        else:
            path.pop()


def add_To_Graph(ax, graph, newPoints, point):
    if len(newPoints) > 1: # If there is anything to add to the graph
        for p in range(len(newPoints) - 1):
            nearest = [ nearest for nearest in graph if (nearest[0] == [ newPoints[p][0], newPoints[p][1] ]) ]
            nearest[0][1].append(newPoints[p + 1])
            graph.append((newPoints[p + 1], []))

            if not p == 0:
                ax.plot(newPoints[p][0], newPoints[p][1], '+k') # First point is already painted
            
            ax.plot([ newPoints[p][0], newPoints[p+1][0] ], [ newPoints[p][1], newPoints[p+1][1] ], color='k', linestyle='-', linewidth=1)

        if point in newPoints:
            ax.plot(point[0], point[1], '.g') # Last point is green
        else:
            ax.plot(newPoints[p + 1][0], newPoints[p + 1][1], '+k') # Last point is not green


def rapidlyExploringRandomTree(ax, img, start, goal, seed=None):
    hundreds = 100
    random.seed(seed)
    points = []
    graph = []
    points.append(start)
    graph.append((start, []))
    print ('Generating and conecting random points')
    occupied = True
    phaseTwo = False

  # Phase two values (points 5 step distances around the goal point)
    minX = max(goal[0] - 5 * STEP_DIS, 0)
    maxX = min(goal[0] + 5 * STEP_DIS, len(img[0]) - 1)
    minY = max(goal[1] - 5 * STEP_DIS, 0)
    maxY = min(goal[1] + 5 * STEP_DIS, len(img) - 1)

    i = 0
    while (goal not in points) and (len(points) < MAX_NUM_VERT):
        if (i % 100) == 0:
            print (i, 'points randomly generated')

        if (len(points) % hundreds) == 0:
            print (len(points), 'vertex generated')
            hundreds = hundreds + 100

        while(occupied):
            if phaseTwo and (random.random() > 0.8):
                point = [ random.randint(minX, maxX), random.randint(minY, maxY) ]
            else:
                point = [ random.randint(0, len(img[0]) - 1), random.randint(0, len(img) - 1) ]

            if(img[point[1]][point[0]][0] * 255 == 255):
                occupied = False

        occupied = True

        nearest = find_Nearest_Point(points, point)
        newPoints = connect_Points(point, nearest, img)
        add_To_Graph(ax, graph, newPoints, point)
        newPoints.pop(0) # The first element is already in the points list
        points.extend(newPoints)
        ppl.draw()
        i = i + 1

        if len(points) >= MIN_NUM_VERT:
            if not phaseTwo:
                print ('Phase Two')
        phaseTwo = True

        if phaseTwo:
            nearest = find_Nearest_Point(points, goal)
            newPoints = connect_Points(goal, nearest, img)
            add_To_Graph(ax, graph, newPoints, goal)
            newPoints.pop(0)
            points.extend(newPoints)
            ppl.draw()

        if goal in points:
            print ('Goal found, total vertex in graph:', len(points), 'total random points generated:', i)
            path = searchPath(graph, start, [start])

            for i in range(len(path)-1):
                ax.plot([ path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1] ], color='g', linestyle='-', linewidth=2)
                ppl.draw()

            print ('Showing resulting map')
            print ('Final path:', path)
            print ('The final path is made from:', len(path),'connected points')
        else:
            path = None
            print ('Reached maximum number of vertex and goal was not found')
            print ('Total vertex in graph:', len(points), 'total random points generated:', i)
            print ('Showing resulting map')

    ppl.show()
    return path

def map_builder():
    fixed_obstacle = {
        'B1': [ 7.08, 1.00, 8.08, 1.2],
        'B2': [ 5.78, 2.14 ,6.58 , 2.34], 
        'B3': [ 6.58, 3.48, 6.78, 4.48],
        'B4': [ 3.54, 0.935, 4.45 , 1.135], 
        'B5': [ 3.864, 2.064, 4.216, 2.416], 
        'B6': [ 3.54, 3.345, 4.45 , 3.545],
        'B7': [ 1.5, 0, 1.7, 1],
        'B8': [ 1.5, 2.14, 2.3, 2.34], 
        'B9': [ 0, 3.28, 1, 3.48],
        'B10': [ 0, 0, 0.02, 4.48 ],
        'B11': [ 8.08, 0, 8.1, 4.48],
        'B12': [ 0, 0, 8.1, 0.02 ],
        'B13': [ 0, 4.48, 8.08, 4.50 ]
    }

    ox1, oy1, ox2, oy2 = [], [], [], []

    for name in fixed_obstacle:
        ox1.append( fixed_obstacle[name][0] / 0.02 )
        oy1.append( fixed_obstacle[name][1] / 0.02 )
        ox2.append( fixed_obstacle[name][2] / 0.02 )
        oy2.append( fixed_obstacle[name][3] / 0.02 )

    obstacles = list(zip(ox1, oy1, ox2, oy2))
    # 8.08 / 0.02 = 404
    # 4.48 / 0.02 = 224
    obstacle_map = np.full( ( 405, 225 ), 255 )

    for pos in obstacles:
        for x in np.arange(pos[0], pos[2]):
            for y in np.arange(pos[1], pos[3]):    
                obstacle_map[ (int)( x ), (int)( y ) ] = 0

    return obstacle_map


ori_obstacle_map = map_builder()
g_obstacle_map = map_builder()

def update_map(obstacle):
    # print( obstacle )
    global g_obstacle_map
    g_obstacle_map = ori_obstacle_map
    for( xx, yy ) in obstacle:
        for x in np.arange( xx / 0.02 , xx / 0.02  ):
            for y in np.arange( yy / 0.02 , yy / 0.02  ):    
                g_obstacle_map[ (int)( x ), (int)( y ) ] = 0
