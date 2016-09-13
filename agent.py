import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import namedtuple
import pprint as pp

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.learningAgentState = namedtuple(
            'learningAgentState',
                # ['light', 'light_violation', 'row', 'next_waypoint', 'hurry'])
                ['light', 'light_violation', 'row', 'next_waypoint'])
        self.actions = [None, 'forward', 'left', 'right']
        self.qTable = {}
        self.logfile = open('c:\\agent_log_file'+'.txt', 'w')
        self.num_moves = 0
        self.reach_dest = 0
        self.current_trial = 0
        self.penalty = 0
        self.n_trials = 100

        self.alpha = .8
        self.gamma = .9
        self.e = .7


    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.state = self.learningAgentState(light=None, light_violation=None,
            row=None, next_waypoint=None)
            # hurry=None)

    @staticmethod
    def has_right_of_way(inputs, next_waypoint):
        """A static function to check if the aent has right-of-way."""
        if (next_waypoint == 'left' and inputs['light'] == 'green' and inputs['oncoming'] == 'forward'):
            return 'no_left'
        elif (next_waypoint == 'right' and inputs['light'] == 'red' and
            (inputs['oncoming'] == 'left' or inputs['left'] == 'forward')):
            return 'no_right'
        else:
            return 'has_row'

    @staticmethod
    def is_violating_light(light, next_waypoint):
        """A static function to check if the agent is violating the traffic signal light."""
        if (light == 'red' and next_waypoint is not 'right'):
            return 1
        else:
            return 0

    @staticmethod
    def is_late(deadline):
        """A static function to check if agent is running out of moves."""
        if deadline < 5:
            return 1
        else:
            return 0

    def to_action(self, x):
        """Converts the location of a provided list to its corresponding action."""
        return {
        '0': None,
        '1': 'forward',
        '2': 'left',
        '3': 'right'
    }[x]

    def to_pos(self, x):
        """Conversts an action to its corresponding location in a list."""
        return {
        'None': 0,
        'forward': 1,
        'left': 2,
        'right': 3
    }[x]

    def beOptimistic(self, state):
        """Modifies the value of an action to encourage the agent to try it."""
        values = self.qTable[state]
        m = max(values)
        for i, v in enumerate(values):
            if (v == 0):
                self.qTable[state][i] = m + .05
                break

    def get_action(self, state):
        """Determines the action of the agent, either try new action or choose best previous action."""
        try:
            k=state.index(0)
            return self.to_action( str(k) )
        except ValueError:
            return self.to_action(str(state.index(max(state))))

    # def get_action(self, state):
    #     curr_e = self.e * ( float(self.n_trials - self.current_trial)/self.n_trials )
    #     if random.random() > curr_e:
    #         return self.to_action(str(state.index(max(state))))
    #     else:
    #         return random.choice(self.actions)

    def get_success_rate(self):
        return "{}/{} = %{}".format(self.reach_dest,
            self.current_trial, (round(float(self.reach_dest)/float(self.current_trial), 3))*100)

    def get_penalty_ratio(self):
        return "{}/{} = %{}".format(self.penalty, self.num_moves,
            (round(float(self.penalty)/float(self.num_moves),4))*100)

    def get_parameters(self):
        return self.alpha, self.gamma, self.e

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        self.num_moves += 1
        decayingAlpha = self.alpha/self.num_moves

        if t == 0:
            self.current_trial += 1

        # Update state
        self.state = self.learningAgentState(light=inputs['light'],
            light_violation=self.is_violating_light(inputs['light'], self.next_waypoint),
            row=self.has_right_of_way(inputs, self.next_waypoint),
            next_waypoint=self.next_waypoint)
            # hurry=self.is_late(deadline))


        # Select action according to your policy
        if self.state not in self.qTable:
            # action = random.choice(self.actions)
            action = self.next_waypoint
            # Initialize new state
            self.qTable[self.state] = [0, 0, 0, 0]
        else:
            sa = self.qTable[self.state]
            # action = self.to_action(str(sa.index(max(sa))))
            action = self.get_action(self.qTable[self.state])

        # Execute action and get reward
        reward = self.env.act(self, action)
        # if reward > 5:
            # reward = reward - 10

        if reward >= 10:
            self.reach_dest += 1
        elif reward < 10 and reward > 5:
            self.reach_dest += 1
            self.penalty += 1
        elif reward < 0:
            self.penalty += 1


        # success_rate = "{}/{} = %{}".format(self.reach_dest,
        #     self.current_trial, (round(float(self.reach_dest)/float(self.current_trial), 3))*100)
        #
        # penalty_ratio = "{}/{} = %{}".format(self.penalty, self.num_moves,
        #     (round(float(self.penalty)/float(self.num_moves),4))*100)
        #
        # pp.pprint(success_rate, self.logfile)
        # pp.pprint(penalty_ratio, self.logfile)

        # Sense environment after action/reward
        inputs_2 = self.env.sense(self)
        deadline_2 = self.env.get_deadline(self)
        next_waypoint_2 = self.planner.next_waypoint()

        # store new state
        state_2 = self.learningAgentState(light=inputs_2['light'],
            light_violation=self.is_violating_light(inputs_2['light'], next_waypoint_2),
            row=self.has_right_of_way(inputs_2, next_waypoint_2),
            next_waypoint=next_waypoint_2)
            # hurry=self.is_late(deadline_2))

        # Initialize next new state
        if state_2 not in self.qTable:
            self.qTable[state_2] = [0, 0, 0, 0]

        # Formula for updating Q-Table
        qsa = (1 - decayingAlpha) * self.qTable[self.state][self.to_pos(str(action))] + decayingAlpha * (reward + self.gamma * max(self.qTable[state_2]))

        # Update Q-Table
        self.qTable[self.state][self.to_pos(str(action))] = qsa

        # self.beOptimistic(self.state)

        # pp.pprint("\n", self.logfile)
        # pp.pprint("LearningAgent.update(): deadline = {}, inputs = {}, action = {}, next_waypoint = {}, reward = {}".format(deadline, inputs, action, self.next_waypoint, reward), self.logfile)  # [debug]
        # pp.pprint(self.qTable, self.logfile)
        # pp.pprint(t, self.logfile)
        # pp.pprint("\n", self.logfile)
        # TODO: Learn policy based on state, action, reward
        #if inputs['oncoming'] is not None or inputs['right'] is not None or inputs['left'] is not None:
        #    print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, next_waypoint = {}, reward = {}".format(deadline, inputs, action, self.next_waypoint, reward)  # [debug]
        #    print "Is violating light = {}".format(self.is_violating_light(inputs['light'], self.next_waypoint))
        #    print "Has right of way = {}".format(self.has_right_of_way(inputs, self.next_waypoint))
        #    print "Light color = {}".format(inputs['light'])
        #    print "Should hurry = {}".format(self.is_late(deadline))
        #    print "\n"
        #print self.is_conflict(inputs, self.next_waypoint)


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.001)  # reduce update_delay to speed up simulation
    sim.run(a.n_trials)  # press Esc or close pygame window to quit

    # pp.pprint("Success rate: ", a.logfile)
    # pp.pprint(a.get_success_rate(), a.logfile)
    # pp.pprint("Penalty ratio: ", a.logfile)
    # pp.pprint(a.get_penalty_ratio(), a.logfile)
    # pp.pprint("Parameters alpha, gamma, and epsilon: ", a.logfile)
    # pp.pprint(a.get_parameters(), a.logfile)
    # pp.pprint("Number of states explored: ", a.logfile)
    # pp.pprint(len(a.qTable), a.logfile)
    # pp.pprint("Q-Table: ", a.logfile)
    # pp.pprint(a.qTable, a.logfile)



if __name__ == '__main__':
    run()
