from collections import deque
import heapq

def parse_file(file_name):
    with open(file_name,"r") as f:
        input=f.readlines()
        method=input[0].strip()
        energy_limit=int(input[1].strip())
        n=int(input[2])
        map=RoverMap()
        for i in range(3,n+3):
            st=input[i].strip().split()
            node_name,x,y,z=st[0],int(st[1]),int(st[2]),int(st[3])
            map.add_nodes(node_name,x,y,z)
        e=int(input[n+3])
        for i in range(n+4,n+e+4):
            ed=input[i].strip().split()
            from_node,to_node=ed
            map.add_edges(from_node,to_node)
    return method,energy_limit,map

class RoverMap:
    def __init__(self):
        self.nodes={}
        self.edges={}
    
    def add_nodes(self,node_name,x,y,z):
        self.nodes[node_name]=[x,y,z]
    
    def add_edges(self,from_node,to_node):
        if from_node in self.edges:
            self.edges[from_node].append(to_node)
        else:
            self.edges[from_node]=[to_node]
        
        if to_node in self.edges:
            self.edges[to_node].append(from_node)
        else:
            self.edges[to_node]=[from_node]


class BFS:
    def __init__(self,energy_limit,map):
        self.start='start'
        self.goal='goal'
        self.energy_limit=energy_limit
        self.map=map
    
    def runBFS(self):
        q = deque([(self.start, "", '')])
        visited_edges=set()
        while q:
            current_state, path, parent = q.popleft()
            if len(parent)==0:
                parent=None
            for adj in self.map.edges[current_state]:
                edge = f'{current_state} {adj}'
                if edge not in visited_edges and self.check_energy_required(parent, current_state, adj):
                    if adj == self.goal:
                        return path + " " + current_state + " " + self.goal
                    visited_edges.add(edge)
                    q.append((adj, path + " " + current_state, current_state))
        return None
    
    def check_energy_required(self,parent,current_state,adj):
        energy_required = self.map.nodes[adj][2] - self.map.nodes[current_state][2]
        momentum = None
        if parent is None:
            momentum = 0
        else:
            mom_calc=self.map.nodes[parent][2]- self.map.nodes[current_state][2]
            if  mom_calc>0:
                momentum= mom_calc
            else:
                momentum=0
        return momentum + self.energy_limit >= energy_required

class UCS:
    def __init__(self,energy_limit,map):
        self.start='start'
        self.goal='goal'
        self.energy_limit=energy_limit
        self.map=map
    def runUCS(self):
        open_q=[]
        visited_edges={}
        #cummulative cost, current node, path,parent
        heapq.heappush(open_q,(0,self.start,"",'')) 
        while open_q:
            cumm_cost,current_state,path,parent=heapq.heappop(open_q)
            if len(parent)==0:
                parent=None
            
            if current_state=='goal':
                return path+" "+'goal'
            
            prev_edge=f'{parent} {current_state}'
            if prev_edge in visited_edges:
                if visited_edges[prev_edge][-1]<cumm_cost:
                    continue
            else:
                visited_edges[prev_edge]=(path,cumm_cost)
            
            for adj in self.map.edges[current_state]:
                xa,ya=self.map.nodes[current_state][:2]
                xb,yb=self.map.nodes[adj][:2]
                new_path_cost=((xa-xb)**2 + (ya-yb)**2)**0.5
                updated_cumm_cost=cumm_cost+new_path_cost
                if self.check_energy_required(parent,current_state,adj):
                    adj_edge=f'{current_state} {adj}'
                    if adj_edge not in visited_edges : 
                        heapq.heappush(open_q,(updated_cumm_cost,adj,path+" "+current_state,current_state))
                    elif visited_edges[adj_edge][-1] > updated_cumm_cost:
                        heapq.heappush(open_q,(updated_cumm_cost,adj,path+" "+current_state,current_state))
        return None

    def check_energy_required(self,parent,current_state,adj):
        energy_required = self.map.nodes[adj][2] - self.map.nodes[current_state][2]
        momentum = None
        if parent is None:
            momentum = 0
        else:
            mom_calc=self.map.nodes[parent][2]- self.map.nodes[current_state][2]
            if  mom_calc>0:
                momentum= mom_calc
            else:
                momentum=0
        return momentum + self.energy_limit >= energy_required
    
        
class Astar:
    def __init__(self,energy_limit,map):
        self.start='start'
        self.goal='goal'
        self.energy_limit=energy_limit
        self.map=map
    def runAstar(self):
        open_q=[]
        visited_edges={}
        #cummulative traversal path cost, heuristic cost, current node, path, parent
        heapq.heappush(open_q,(0,0,self.start,"",''))
        while open_q:
            cumm_tp_cost,heuristic_cost,current_state,path,parent=heapq.heappop(open_q)
            if len(parent)==0:
                parent=None
            
            if current_state=='goal':
                return path+" "+'goal'
            
            prev_edge=f'{parent} {current_state}'
            cumm_tp_cost=cumm_tp_cost-heuristic_cost
            if prev_edge in visited_edges:
                if visited_edges[prev_edge][-1]<cumm_tp_cost:
                    continue
            else:
                visited_edges[prev_edge]=(path,cumm_tp_cost)
            
            for adj in self.map.edges[current_state]:
                if self.check_energy_required(parent,current_state,adj):
                    xa,ya,za=self.map.nodes[current_state][:]
                    xb,yb,zb=self.map.nodes[adj][:]
                    current_adj_path_cost=((xa-xb)**2 + (ya-yb)**2 + (za-zb)**2)**0.5
                    heuristic_adj_goal=self.calc_heuristic_cost(adj)
                    
                    updated_cumm_cost=cumm_tp_cost+current_adj_path_cost
                    adj_edge=f'{current_state} {adj}'
                    if adj_edge not in visited_edges:
                        heapq.heappush(open_q,(updated_cumm_cost+heuristic_adj_goal,heuristic_adj_goal,adj,path+" "+current_state,current_state))
                    elif (visited_edges[adj_edge][-1] > updated_cumm_cost):
                        heapq.heappush(open_q,(updated_cumm_cost+heuristic_adj_goal,heuristic_adj_goal,adj,path+" "+current_state,current_state))
        return None

    def calc_heuristic_cost(self,adjacent):
        #For an admissible heuristic, h(goal)=0 and h(n) will never over estimate true value.
        #considering euclidean distance over manhattan distance as the heuristic function as it's appropriate for the given problem.
        xg,yg,zg=self.map.nodes[self.goal][:]
        xa,ya,za=self.map.nodes[adjacent][:]
        return ((xa-xg)**2+(ya-yg)**2+(za-zg)**2)**0.5

    def check_energy_required(self,parent,current_state,adj):
        energy_required = self.map.nodes[adj][2] - self.map.nodes[current_state][2]
        momentum = None
        if parent is None:
            momentum = 0
        else:
            mom_calc=self.map.nodes[parent][2]- self.map.nodes[current_state][2]
            if  mom_calc>0:
                momentum= mom_calc
            else:
                momentum=0
        return momentum + self.energy_limit >= energy_required

def run():
    method,energy_limit,map=parse_file('training-v2/input7.txt')#change this
    if method== 'BFS':
        bfs=BFS(energy_limit,map)
        path=bfs.runBFS()
    elif method=='UCS':
        ucs=UCS(energy_limit,map)
        path=ucs.runUCS()
    elif method=='A*':
        astar=Astar(energy_limit,map)
        path=astar.runAstar()
    
    if path is None:
        with open("output1.txt", 'w') as op:
            op.write("FAIL")
    else:
        with open("output1.txt",'w') as op:
            op.write(path.strip())

if __name__ == '__main__':
    run()