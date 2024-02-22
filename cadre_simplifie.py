# -*- coding: utf-8 -*-

from numpy import random
import numpy as np
import matplotlib.pyplot as plt
import imageio
import time
from ast import literal_eval
import os
# Recuperer le chemin du fichier
os.chdir(os.path.dirname(os.path.abspath(__file__)))





class Position:

    def __init__(self, north, east, south, west):
        self.north = north
        self.east = east
        self.south = south
        self.west = west
        self.walls = [north, east, south, west]
        self.exit = False
        self.player = False

    def get_walls(self):
        return self.walls

    def set_walls(self, new_walls):
        self.walls = new_walls

    def actualize_walls(self):
        self.set_walls([self.north, self.east, self.south, self.west])

    def set_north(self, north):
        self.north = north
        self.actualize_walls()

    def set_east(self, east):
        self.east = east
        self.actualize_walls()

    def set_south(self, south):
        self.south = south
        self.actualize_walls()

    def set_west(self, west):
        self.west = west
        self.actualize_walls()

    def place_player(self):
        self.player = True

    def remove_player(self):
        self.player = False

    def set_exit(self):
        self.exit = True

    def remove_exit(self):
        self.exit = False



class Maze:

    def __init__(self, size=10):
        if size < 3:
            print("taille trop petite")
            raise ValueError
        else:
            self.size = size
            self.all_positions = [[Position(j == 0,
                                            i == size - 1,
                                            j == size - 1,
                                            i == 0)
                                   for i in range(size)] for j in range(size)]
            self.exit = None
            self.player = None
            self.positions_explored = []

    def randomize(self):
        self.remove_exit()
        self.remove_player()
        x = int(random.random()*self.size)
        y = int(random.random()*self.size)
        self.set_exit(x, y)
        xj = x
        yj = y
        while xj == x and yj == y:
            xj = int(random.random()*self.size)
            yj = int(random.random()*self.size)
        self.place_player(xj, yj)
        
        for i in range(self.size):
            for j in range(self.size):
                if i != self.size -1:
                    self.changeWall(i, j, 2, int(random.random()*2))
                if j != self.size - 1:
                    self.changeWall(i, j, 1, int(random.random()*2))

    def get_position(self, i, j):
        return self.all_positions[i][j]

    def __str__(self):
        maze_str = " "
        # first line
        for i in range(self.size):
            maze_str += "__"
        maze_str += "\n"
        for i in range(self.size):
            for j in range(self.size):
                position = self.get_position(i, j)
                walls = position.get_walls()
                if walls[3]:
                    maze_str += "|"
                    if walls[2]:
                        if position.exit:
                            maze_str += "\033[42;m_\033[0m"
                        elif position.player:
                            maze_str += "\033[41;m_\033[0m"
                        else:
                            maze_str += "_"
                    else:
                        if position.exit:
                            maze_str += "\033[42;m \033[0m"
                        elif position.player:
                            maze_str += "\033[41;m \033[0m"
                        else:
                            maze_str += " "
                elif walls[2]:
                    if position.exit:
                        maze_str += "\033[42;m__\033[0m"
                    elif position.player:
                        maze_str += "\033[41;m__\033[0m"
                    else:
                        maze_str += "__"
                else:
                    if position.exit:
                        maze_str += " "
                        maze_str += "\033[42;m \033[0m"
                    elif position.player:
                        maze_str += " "
                        maze_str += "\033[41;m \033[0m"
                    else:
                        maze_str += "  "
                if j == self.size - 1:
                    if i == self.size - 1:
                        maze_str += "|\n"
                    else:
                        maze_str += "|\n"
        return maze_str

    def plot(self, filename = None):
        n = self.size
        plt.figure(figsize = (10, 10))
        plt.plot([0,0,n,n,0],[0,n,n,0,0],color = 'black',linewidth=7.0)
      

        for i in range(10):
            plt.plot([i,i],[0,n],color = 'black',alpha = 0.2)
            plt.plot([0,n],[i,i],color = 'black',alpha = 0.2)

        for i in range(n):
            for j in range(n):
                if self.get_position(j, i).east:
                    plt.plot([i+1,i+1],[10-j,10-j-1],linewidth=7.0,color = 'black')
                if self.get_position(j, i).south:
                    plt.plot([i,i+1],[10-j-1,10-j-1],linewidth=7.0,color = 'black')
        
        try:
            (j,i) = self.player
            j += 0.5
            i += 0.5
            plt.scatter(i,10-j,linewidths = 5,color = 'red')
            
            (j,i) = self.exit
            j += 0.5
            i += 0.5
            plt.scatter(i,10-j,linewidths = 30,color = 'green',marker = '+')
        except:
            pass
        
        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()    
        plt.close()



    def changeWall(self, i, j, wallNumber, wallValue):
        position = self.get_position(i, j)
        if wallNumber == 0:
            position.set_north(wallValue)
            if i != 0:
                self.get_position(i - 1, j).set_south(wallValue)
        elif wallNumber == 1:
            position.set_east(wallValue)
            if j != self.size - 1:
                self.get_position(i, j + 1).set_west(wallValue)
        elif wallNumber == 2:
            position.set_south(wallValue)
            if i != self.size - 1:
                self.get_position(i + 1, j).set_north(wallValue)
        elif wallNumber == 3:
            position.set_west(wallValue)
            if j != 0:
                self.get_position(i, j - 1).set_east(wallValue)

    def set_exit(self, i, j):
        if self.exit is None:
            self.get_position(i, j).set_exit()
            self.exit = (i, j)

    def remove_exit(self):
        if self.exit is not None:
            self.get_position(*self.exit).remove_exit()
            self.exit = None

    def place_player(self, i, j):
        if self.player is None:
            self.get_position(i, j).place_player()
            self.player = (i, j)

    def remove_player(self):
        if self.player is not None:
            self.get_position(*self.player).remove_player()
            self.player = None

    def move_player(self, i, j, new_i, new_j, force = False):
        if force:
            self.remove_player()
            self.place_player(new_i, new_j)
            self.positions_explored = []
            return True
        else:
            flag = False
            walls = self.get_position(i, j).get_walls()
            if new_i == i + 1:
                if not walls[2]:
                    flag = True
            elif new_i == i - 1:
                if not walls[0]:
                    flag = True
            elif new_j == j + 1:
                if not walls[1]:
                    flag = True
            elif new_j == j - 1:
                if not walls[3]:
                    flag = True
            if flag:
                self.positions_explored.append((self.player))
                self.remove_player()
                self.place_player(new_i, new_j)
                return True
            else:
                return False

    def environment(self, i, j):
        return Environment(self.get_position(i, j).walls,
                           self.player, self.exit,
                           self.positions_explored)

class Environment:

    def __init__(self, walls, player_position, exit_position, explored):
        self.walls = walls
        self.player = player_position
        self.exit = exit_position
        self.explored = explored

    def copy(self):
        playerc = (self.player[0], self.player[1])
        exitc = (self.exit[0], self.exit[1])
        c = Environment(self.walls.copy(), playerc, exitc, self.explored.copy())
        return c

class Action:

    def __init__(self, direction):
        self.direction = direction

    def copy(self):
        return Action(self.direction)

class Trajectory:

    def __init__(self, tab = []):
        self.trajectory = tab

    def save(self, fileName="test.ty"):
        self.writeOrAppend(fileName, "w")

    def append(self, fileName="test.ty"):
        self.writeOrAppend(fileName, "a")

    def writeOrAppend(self, fileName, mode):
        with open(fileName, mode) as f:
            f.write(str(len(self.trajectory)))
            f.write("\n")
            for segment in self.trajectory:
                f.write(str(segment[0].walls))
                f.write("\n")
                f.write(str(segment[0].player))
                f.write("\n")
                f.write(str(segment[0].exit))
                f.write("\n")
                f.write(str(segment[1].direction))
                f.write("\n")
                f.write(str(len(segment[0].explored)))
                f.write("\n")
                for k in range(len(segment[0].explored)):
                    f.write(str(segment[0].explored[k]))
                    f.write("\n")
                f.write("\n")
            f.write("\n")

    def readAll(self, fileName="test.ty"):
        with open(fileName, "r") as f:
            lines = f.readlines()
            nbrSegment = int(int(lines[0]))
            self.trajectory = []
            for k in range(nbrSegment):
                walls = lines[5*k + 1]
                player_position = lines[5*k + 2]
                exit_position = lines[5*k + 3]
                direction = lines[5*k + 4]
                env = Environment(literal_eval(walls),
                                  literal_eval(player_position),
                                  literal_eval(exit_position))
                act = Action(int(direction))
                self.trajectory.append((env, act))

    def get_mus(self, fileName="mus.ty"):
        with open(fileName, "r") as f:
            for mu in f.readlines():
                self.mus.append(float(mu))

    def plot(self, maze):
        init_player = maze.player
        print(maze)
        maze.plot()
        time.sleep(2)
        nbr_action = len(self.trajectory)
        print(nbr_action)
        
        for segment in range(nbr_action):
            direction = self.trajectory[segment][1].direction
            print(direction)
            print(f"segment : {segment+1}/{nbr_action}")
            if direction == 0:
                print("trying to go up")
                maze.move_player(*maze.player, maze.player[0] - 1, maze.player[1])
            if direction == 1:
                print("trying to go right")
                maze.move_player(*maze.player, maze.player[0], maze.player[1] + 1)
            if direction == 2:
                print("trying to go down")
                maze.move_player(*maze.player, maze.player[0] + 1, maze.player[1])
            if direction == 3:
                print("trying to go left")
                maze.move_player(*maze.player, maze.player[0], maze.player[1] - 1)
            
            print(maze)
            maze.plot()
            time.sleep(2)
        maze.move_player(*maze.player, *init_player, force = True)


class Preference:

    def __init__(self, trajectory1, trajectory2, mu1, mu2):
        self.trajectory1 = trajectory1
        self.trajectory2 = trajectory2
        self.mu1 = mu1
        self.mu2 = mu2

    def save(self, fileName = "preference.ty", mode = "w"):
        self.trajectory1.writeOrAppend(fileName, mode)
        self.trajectory2.writeOrAppend(fileName, "a")
        with open(fileName, "a") as f:
            f.write(str(self.mu1))
            f.write("\n")
            f.write(str(self.mu2))
            f.write("\n")
            f.write("\n")

    def compute_probability(self, function):
        r1 = sum([function(k) for k in self.trajectory1.trajectory])
        r2 = sum([function(k) for k in self.trajectory2.trajectory])
        return (np.exp(r1))/(np.exp(r1) + np.exp(r2))

    def compute_loss(self, function):
        proba = self.compute_probability(function)
        return self.mu1*np.log(proba) + self.mu2*np.log(1-proba)

class ListPreference:

    def __init__(self, L=[]):
        self.preferences = L

    def read(self, fileName = "preferences.ty"):
        self.preferences = []
        with open(fileName, "r") as f:
            lines = f.readlines()
            actualLine = 0
            while actualLine < len(lines):
                trajectory1 = []
                actualLineUpdate = 0
                actualLine += 1
                for k in range(int(lines[actualLine - 1])):
                    walls = lines[actualLine]
                    actualLine += 1
                    player_position = lines[actualLine]
                    actualLine += 1
                    exit_position = lines[actualLine]
                    actualLine += 1
                    direction = lines[actualLine]
                    actualLine += 1
                    l = int(lines[actualLine])
                    actualLine += 1
                    explored = []
                    for _ in range(l):
                        explored.append(literal_eval(lines[actualLine]))
                        actualLine += 1
                    actualLine += 1
                    env = Environment(literal_eval(walls),
                                      literal_eval(player_position),
                                      literal_eval(exit_position),
                                      explored)
                    act = Action(int(direction))
                    trajectory1.append((env, act))
                actualLine += actualLineUpdate + 2
                trajectory2 = []
                actualLineUpdate = 0
                for k in range(int(lines[actualLine - 1])):
                    walls = lines[actualLine]
                    actualLine += 1
                    player_position = lines[actualLine]
                    actualLine += 1
                    exit_position = lines[actualLine]
                    actualLine += 1
                    direction = lines[actualLine]
                    actualLine += 1
                    l = int(lines[actualLine])
                    actualLine += 1
                    explored = []
                    for _ in range(l):
                        explored.append(literal_eval(lines[actualLine]))
                        actualLine += 1
                    actualLine += 1
                    env = Environment(literal_eval(walls),
                                      literal_eval(player_position),
                                      literal_eval(exit_position),
                                      explored)
                    act = Action(int(direction))
                    trajectory2.append((env, act))
                actualLine += actualLineUpdate + 1
                
                mu1 = lines[actualLine]
                actualLine += 1
                mu2 = lines[actualLine]
                actualLine += 1
                
                self.preferences.append(Preference(Trajectory(trajectory1),
                                                   Trajectory(trajectory2),
                                                   float(mu1), float(mu2)))
                
                actualLine += 1

    def compute_loss(self, function):
        return -sum([k.compute_loss(function) for k in self.preferences])


class AI:

    def __init__(self):
        self.list = ListPreference()
        self.param_function = None
        self.best_trajectory = None

    def read(self, fileName):
        self.list.read(fileName)

    def get_preferences(self, fileName = "preferences.ty"):
        self.list.read(fileName)

    def function(self, sigma, alpha, beta, gamma):
        deplacement = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        return (2*alpha*sum([np.abs(sigma[0].player[k] - sigma[0].exit[k]) for k in range(2)]) +
                    3*beta*sigma[0].walls[sigma[1].direction] -
                    gamma * sum([sum([np.abs(sigma[0].explored[m][k] - sigma[0].player[k] - deplacement[sigma[1].direction][k]) for k in range(2)]) for m in range(len(sigma[0].explored))]))

    def max(self, tab):
        if len(tab) == 0:
            return 0
        else:
            return max(tab)

    def get_loss(self, alpha, beta, gamma):
        if alpha + beta + gamma != 1:
            return 1000
        else:
            return self.list.compute_loss(lambda sigma: self.function(sigma, alpha, beta, gamma))

    def min_loss(self, alpha_list, beta_list, gamma_list):
        m_loss = 100_000
        min_alpha = None
        min_beta = None
        min_gamma = None
        for alpha in alpha_list:
            for beta in beta_list:
                for gamma in gamma_list:
                    comput = self.get_loss(alpha, beta, gamma)
                    if comput < m_loss:
                        m_loss = comput
                        min_alpha = alpha
                        min_beta = beta
                        min_gamma = gamma
        self.param_function = [min_alpha, min_beta, min_gamma]

    def ask_preference(self, maze, nbr_move, fileName):
        both_traj = []
        traj1 = self.choose_trajectory1(nbr_move, maze)
        traj2 = self.choose_trajectory2(nbr_move, maze)
        both_traj = [traj1, traj2]
        print("first trajectory :")
        both_traj[0].plot(maze)
        print("second trajectory :")
        both_traj[1].plot(maze)
        print("which one do you prefer ? (1 or 2)")
        print("If you consider that both segments are equally preferable, type 3.")
        print("If you consider that the segments are not comparable, type 0.")
        choice = 4
        while choice != 0 and choice != 1 and choice != 2 and choice != 3:
            choice = int(input("Please type 0, 1, 2 or 3 : "))

        if choice == 1:
            new_pref = Preference(both_traj[0], both_traj[1], 1, 0)
        elif choice == 2:
            new_pref = Preference(both_traj[0], both_traj[1], 0, 1)
        elif choice == 3:
            new_pref = Preference(both_traj[0], both_traj[1], 1/2, 1/2)
        elif choice == 0:
            print("Thank you!")
            return
        if len(self.list.preferences) == 0:
            new_pref.save(fileName, "w")
        else:
            new_pref.save(fileName, "a")
        self.list.preferences.append(new_pref)

    def choose_trajectory1(self, nbr_move, maze):
        if len(self.list.preferences) == 0:
            return self.random_trajectory(nbr_move, maze)
        else:
            if self.best_trajectory is None:
                if self.param_function is None:
                    self.min_loss(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01), np.arange(0, 1, 0.01))
                self.get_best_trajectory(nbr_move, maze)
            return self.best_trajectory1

    def choose_trajectory2(self, nbr_move, maze):
        print(len(self.list.preferences))
        if len(self.list.preferences) < 100:
            return self.random_trajectory(nbr_move, maze)
        else:
            if self.best_trajectory is None:
                if self.param_function is None:
                    self.min_loss(np.linspace(1, 10, 100), np.linspace(1, 10, 100), np.arange(0, 1, 0.01))
                self.get_best_trajectory(nbr_move, maze)
            return self.best_trajectory2


    def random_trajectory(self, nbr_move, maze):
        traj = []
        init_player = maze.player
        for _ in range(nbr_move):
            env = maze.environment(*maze.player)
            direction = random.randint(4)
            traj.append((env, Action(direction)))
            if direction == 0:
                pos_player = (maze.player[0] - 1, maze.player[1])
            elif direction == 1:
                pos_player = (maze.player[0], maze.player[1] + 1)
            elif direction == 2:
                pos_player = (maze.player[0] + 1, maze.player[1])
            elif direction == 3:
                pos_player = (maze.player[0], maze.player[1] - 1)
            maze.move_player(*maze.player, *pos_player)
        maze.move_player(*maze.player, *init_player, force=True)
        return Trajectory(traj)


    def get_best_trajectory(self, nbr_move, maze):
        r_min_first = 10_000
        best_trajectory_first = None
        r_min_second = 10_000
        best_trajectory_second = None
        init_player = maze.player
        for k in range(4**nbr_move):
            focus = k
            traj = []
            for l in range(nbr_move):
                dire = int(focus/(4**(nbr_move-1-l)))
                focus = focus%(4**(nbr_move-1-l))
                env = maze.environment(*maze.player)
                act = Action(dire)
                traj.append((env.copy(), act.copy()))
                if dire == 0:
                    position_player = (maze.player[0] - 1, maze.player[1])
                elif dire == 1:
                    position_player = (maze.player[0], maze.player[1] + 1)
                elif dire == 2:
                    position_player = (maze.player[0] + 1, maze.player[1])
                elif dire == 3:
                    position_player = (maze.player[0], maze.player[1] - 1)
                maze.move_player(*maze.player, *position_player)
            maze.move_player(*maze.player, *init_player, force=True)
            function_test = sum([self.function(sigma, *self.param_function) for sigma in traj])
            if function_test < r_min_first:
                r_min_first = function_test
                best_trajectory_first = traj
            elif function_test < r_min_second:
                r_min_second = function_test
                best_trajectory_second = traj
        self.best_trajectory1 = Trajectory(best_trajectory_first)
        self.best_trajectory2 = Trajectory(best_trajectory_second)

    def complete(self, maze, nbr_move):
        filename = 'GIF//0.png'
        filenames = [filename]
        maze.plot(filename)
        print(maze)
        deplacement = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        flag = False == (0 == 1)
        count = 0
        while(maze.player != maze.exit and flag):
            trajectory = self.choose_trajectory1(nbr_move, maze)
            # print(self.param_function)
            for k in range(nbr_move):
                maze.move_player(*maze.player, 
                                 maze.player[0] + deplacement[trajectory.trajectory[k][1].direction][0],
                                 maze.player[1] + deplacement[trajectory.trajectory[k][1].direction][1])
                count += 1
                filename = f'GIF//{count}.png'
                filenames.append(filename)
                maze.plot(filename)
                print(maze)
                
                # time.sleep(1)
                if maze.player == maze.exit or count > 100:
                    flag = False
                    break
        if count > 100:
            print("oh no ! The bot has not been trained enough :/")
        else:
            print("Congratulation !!")
            print(f"It made it in {count} moves.")
            return count
            with imageio.get_writer('mygif.gif', mode='I',duration = 3) as writer:
                for filename in filenames:
                    image = imageio.imread(filename)
                    writer.append_data(image)
        
            

def random4():
    rand = random()
    if rand < 1/4:
        return 0
    elif rand < 1/2:
        return 1
    elif rand < 3/4:
        return 2
    elif rand < 1:
        return 3
    else:
        print("error : no position chose !")

def convert_to_move(i, j, n):
    if n == 0:
        return i-1, j
    elif n == 1:
        return i, j+1
    elif n == 2:
        return i+1, j
    elif n == 3:
        return i, j-1
    else:
        print("error : no convert possible")

def maze1():
    maze = Maze()
    for i in range(maze.size):
        maze.changeWall(i, i, 2, True)
        maze.changeWall(i, i, 3, True)
    for i in range(maze.size - 1):
        maze.changeWall(i, i+1, 0, True)
        maze.changeWall(i, i+1, 1, True)
    maze.set_exit(0, 0)
    maze.place_player(maze.size - 1, maze.size - 1)
    return maze

def random_maze():
    maze = Maze()
    n = maze.size
    
    for i in range(n):
        for j in range(n):
            for a in range(4):
                maze.changeWall(i,j,a,1)



    deplacements = [(-1,0),(0,1),(1,0),(0,-1)]
    A = [[i+10*j for i in range(n)] for j in range(n)]
    for i in range(99):  #n*n -1 murs a briser pour tout relier
        (x,y) = (random.randint(0,n),random.randint(0,n)) #selection d'une cellule aleatoire
        a = random.randint(0,4) #mur a ouvrir
        x2 = x + deplacements[a][0]
        y2 = y + deplacements[a][1]
        while not(0<=x2<=9 and 0 <= y2 <= 9 and A[x][y] != A[x2][y2] ):
            (x,y) = (random.randint(0,n), random.randint(0,n)) #selection d'une cellule aleatoire
            a = random.randint(0,4) #mur a ouvrir
            x2 = x + deplacements[a][0]
            y2 = y + deplacements[a][1]        
        maze.changeWall(x,y,a,0)
        
        id_a_changer = A[x2][y2]
        id_remplacement = A[x][y]
        for k in range(n):
            for l in range(n):
                if A[k][l] == id_a_changer:
                    A[k][l] = id_remplacement
    x = random.randint(n)
    y = random.randint(n)
    maze.set_exit(x, y)
    xj = x
    yj = y
    while xj == x and yj == y:
        xj = random.randint(n)
        yj = random.randint(n)
    maze.place_player(xj, yj)
    return maze




def restart_feedback():
    maze = random_maze()
    my_AI = AI()
    my_AI.ask_preference(maze, 4, "preferences.ty")

def add_feedback():
    maze = random_maze()
    my_AI = AI()
    my_AI.read("preferences.ty")
    my_AI.ask_preference(maze, 4, "preferences.ty")

def complete_maze():
    maze = random_maze()
    my_AI = AI()
    my_AI.read("preferences.ty")
    if not os.path.exists("GIF"):
        os.makedirs("GIF")
    my_AI.complete(maze, 7)


if __name__ == "__main__":
    restart_feedback()
