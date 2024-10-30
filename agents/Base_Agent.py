import logging
import os
import sys
import gymnasium as gym
import random
import numpy as np
import torch
import time
# import tensorflow as tf
from nn_builder.pytorch.NN import NN
# from tensorboardX import SummaryWriter
# from torch.optim import optimizer as optim
import torch.optim as optim
from utilities.data_structures.Replay_Buffer import Replay_Buffer
from torch.utils.tensorboard import SummaryWriter
import traci

class Base_Agent(object):

    def __init__(self, config):
        self.logger = self.setup_logger()
        self.debug_mode = config.debug_mode
        # if self.debug_mode: self.tensorboard = SummaryWriter()
        self.config = config
        self.set_random_seeds(config.seed)
        self.environment = config.environment
        self.environment_title = self.get_environment_title()
        self.action_types = "DISCRETE" if self.environment.action_space.dtype == np.int64 else "CONTINUOUS"

        self.lowest_possible_episode_score = self.get_lowest_possible_episode_score()

        self.hyperparameters = config.hyperparameters
        self.average_score_required_to_win = self.get_score_required_to_win()
        self.rolling_score_window = self.get_trials()
        # self.max_steps_per_episode = self.environment.spec.max_episode_steps
        self.total_episode_score_so_far = 0
        self.game_full_episode_scores = []
        self.rolling_results = []
        self.max_rolling_score_seen = float("-inf")
        self.max_episode_score_seen = float("-inf")
        self.env_episode_number = 0
        self.device = config.device
        self.visualise_results_boolean = config.visualise_individual_results
        self.turn_off_exploration = False
        gym.logger.set_level(40)  # stops it from printing an unnecessary warning
        # self.log_game_info()
        self.env_agent_dic=config.environment.utils.agent_dic
        self.relu=torch.nn.ReLU()
        self.intersection_id_size=self.get_intersection_id_size()
        self.intersection_id_embedding_size=self.intersection_id_size
        self.network_embed_size=self.get_network_embed_size()

    def get_environment_title(self):
        """Extracts name of environment from it"""
        try:
            name = self.environment.unwrapped.id
        except AttributeError:
            try:
                if str(self.environment.unwrapped)[1:11] == "FetchReach": return "FetchReach"
                elif str(self.environment.unwrapped)[1:8] == "AntMaze": return "AntMaze"
                elif str(self.environment.unwrapped)[1:7] == "Hopper": return "Hopper"
                elif str(self.environment.unwrapped)[1:9] == "Walker2d": return "Walker2d"
                else:
                    name = self.environment.spec.id.split("-")[0]
            except AttributeError:
                name = str(self.environment.env)
                if name[0:10] == "TimeLimit<": name = name[10:]
                name = name.split(" ")[0]
                if name[0] == "<": name = name[1:]
                if name[-3:] == "Env": name = name[:-3]
        return name

    def get_lowest_possible_episode_score(self):
        """Returns the lowest possible episode score you can get in an environment"""
        if self.environment_title == "Taxi": return -800
        return None

    def get_policy_input_size(self):
        return self.intersection_id_embedding_size+self.network_embed_size
    def get_intersection_id_size(self):
        return self.environment.utils.get_intersection_id_size()
    def get_network_embed_size(self):
        return self.config.network_embed_size

    def get_action_space_size(self,agent_id):
        return self.env_agent_dic[agent_id][1]

    def get_score_required_to_win(self):
        """Gets average score required to win game"""
        print("TITLE ", self.environment_title)
        if self.environment_title == "FetchReach": return -5
        if self.environment_title in ["AntMaze", "Hopper", "Walker2d"]:
            print("Score required to win set to infinity therefore no learning rate annealing will happen")
            return float("inf")
        try: return self.environment.unwrapped.reward_threshold
        except AttributeError:
            try:
                return self.environment.spec.reward_threshold
            except AttributeError:
                return self.environment.unwrapped.spec.reward_threshold

    def get_trials(self):
        """Gets the number of trials to average a score over"""
        if self.environment_title in ["AntMaze", "FetchReach", "Hopper", "Walker2d", "CartPole"]: return 100
        try: return self.environment.unwrapped.trials
        except AttributeError: return self.environment.spec.trials

    def setup_logger(self):
        """Sets up the logger"""
        filename = "Training.log"
        try: 
            if os.path.isfile(filename): 
                os.remove(filename)
        except: pass

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        # create a file handler
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.INFO)
        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(handler)
        return logger

    # def log_game_info(self):
    #     """Logs info relating to the game"""
    #     for ix, param in enumerate([self.environment_title, self.action_types, self.action_size, self.lowest_possible_episode_score,
    #                   self.state_size, self.hyperparameters, self.average_score_required_to_win, self.rolling_score_window,
    #                   self.device]):
    #         self.logger.info("{} -- {}".format(ix, param))

    def set_random_seeds(self, random_seed):
        """Sets all possible random seeds so results can be reproduced"""
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(random_seed)
        # tf.set_random_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
            torch.cuda.manual_seed(random_seed)
        if hasattr(gym.spaces, 'prng'):
            gym.spaces.prng.seed(random_seed)

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        self.environment.seed = self.config.seed
        self.states = self.environment.reset(self.env_episode_number,self.config)

        self.mem_states= None
        self.mem_next_states = None
        self.mem_actions = None
        self.mem_rewards = None
        self.mem_done =None
        self.done = False
        self.total_episode_score_so_far = 0
        self.episode_states = []
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_next_states = []
        self.episode_dones = []
        self.episode_desired_goals = []
        self.episode_achieved_goals = []
        self.episode_observations = []
        if "exploration_strategy" in self.__dict__.keys(): self.exploration_strategy.reset()
        self.logger.info("Reseting game -- New start state {}".format(self.states))

    def track_episodes_data(self):
        """Saves the data from the recent episodes"""
        self.episode_states.append(self.mem_states)
        self.episode_actions.append(self.mem_actions)
        self.episode_rewards.append(self.mem_rewards)
        self.episode_next_states.append(self.mem_next_states)
        self.episode_dones.append(self.mem_done)

    def save_and_print_result(self):
        """Saves and prints results of the game"""
        self.save_result()
        self.print_rolling_result()

    def save_result(self):
        """Saves the result of an episode of the game"""
        self.game_full_episode_scores.append(self.total_episode_score_so_far)
        self.rolling_results.append(np.mean(self.game_full_episode_scores[-1 * self.rolling_score_window:]))
        self.save_max_result_seen()

    def save_max_result_seen(self):
        """Updates the best episode result seen so far"""
        if self.game_full_episode_scores[-1] > self.max_episode_score_seen:
            self.max_episode_score_seen = self.game_full_episode_scores[-1]
            self.max_episode=self.env_episode_number

        if self.rolling_results[-1] > self.max_rolling_score_seen:
            if len(self.rolling_results) > self.rolling_score_window:
                self.max_rolling_score_seen = self.rolling_results[-1]

    def print_rolling_result(self):
        """Prints out the latest episode results"""
        text = """"\r Episode {0}, Score: {3: .2f}, Max score seen @ episode {5: .0f}: {4: .2f}, Rolling score: {1: .2f}, Max rolling score seen: {2: .2f}"""
        sys.stdout.write(text.format(len(self.game_full_episode_scores), self.rolling_results[-1], self.max_rolling_score_seen,
                                     self.game_full_episode_scores[-1], self.max_episode_score_seen, self.max_episode))
        sys.stdout.flush()

    def show_whether_achieved_goal(self):
        """Prints out whether the agent achieved the environment target goal"""
        index_achieved_goal = self.achieved_required_score_at_index()
        print(" ")
        if index_achieved_goal == -1: #this means agent never achieved goal
            print("\033[91m" + "\033[1m" +
                  "{} did not achieve required score \n".format(self.agent_name) +
                  "\033[0m" + "\033[0m")
        else:
            print("\033[92m" + "\033[1m" +
                  "{} achieved required score at episode {} \n".format(self.agent_name, index_achieved_goal) +
                  "\033[0m" + "\033[0m")

    def achieved_required_score_at_index(self):
        """Returns the episode at which agent achieved goal or -1 if it never achieved it"""
        for ix, score in enumerate(self.rolling_results):
            if score > self.average_score_required_to_win:
                return ix
        return -1

    # # def update_learning_rate(self, starting_lr):
    #     """Lowers the learning rate according to how close we are to the solution"""
    #     # TODO: Why this update rule?
    #     if len(self.rolling_results) > 0:
    #         last_rolling_score = self.rolling_results[-1]
    #         if last_rolling_score > 0.75 * self.average_score_required_to_win:
    #             new_lr = starting_lr / 100.0
    #         elif last_rolling_score > 0.6 * self.average_score_required_to_win:
    #             new_lr = starting_lr / 20.0
    #         elif last_rolling_score > 0.5 * self.average_score_required_to_win:
    #             new_lr = starting_lr / 10.0
    #         elif last_rolling_score > 0.25 * self.average_score_required_to_win:
    #             new_lr = starting_lr / 2.0
    #         else:
    #             new_lr = starting_lr
            
    #         for agent in self.agent_dic.values():
    #             for g in agent["optimizer"].param_groups:
    #                 g['lr'] = new_lr

        
    #     if random.random() < 0.001: self.logger.info("Learning rate {}".format(new_lr))

# ----------------------------------------------------------------------------------------------------------------------------
    def run_n_episodes(self, num_episodes=None, show_whether_achieved_goal=True, save_and_print_results=True):
        """Runs game to completion n times and then summarises results and saves model (if asked to)"""
        if num_episodes is None: num_episodes = self.config.num_episodes_to_run
        print("num_episodes: ", num_episodes)
        start = time.time()
        
        self.turn_off_any_epsilon_greedy_exploration()

        if self.config.should_load_model:
            self.load_policies()
            # breakpoint()
        
        while self.env_episode_number < num_episodes:
            # if self.env_episode_number==num_episodes-1:
            #     breakpoint()
            print("episode number: ", self.env_episode_number)
            self.run()
            self.env_episode_number += 1
            if self.config.should_save_model: self.save_policies()
            #if save_and_print_results: self.save_and_print_result()
            self.environment.log_data()


        time_taken = time.time() - start
        if show_whether_achieved_goal: self.show_whether_achieved_goal()
        return self.game_full_episode_scores, self.rolling_results, time_taken

    def run(self):
        """Runs a step within a game including a learning step if required"""
        self.reset_game()
        k = 0
        while not self.done:
            if k % 100 == 0:
                print("traci simulation time: ", traci.simulation.getTime())
            k += 1
            routing_queries=self.environment.get_routing_queries()
            if  self.config.routing_mode=="TTSP" or \
                self.config.routing_mode=="TTSPWRR":
                actions=[0]*len(self.environment.get_routing_queries())
                # creating dummy actions which will later be discarded by the environment
            else:
                actions = self.pick_action(routing_queries)

            
            num_new_exp=self.conduct_action(actions)
            
            if num_new_exp==0:
                continue

            if self.config.training_mode:
                self.save_experience()
                self.learn()
            
    def conduct_action(self, actions):
        """Conducts an action in the environment"""
        self.mem_states,self.mem_actions,self.mem_next_states, self.mem_rewards, self.mem_done, self.done = self.environment.step(actions)
        self.total_episode_score_so_far += sum(self.mem_rewards)
        if self.hyperparameters["clip_rewards"]: self.rewards =  max(min(self.reward, 1.0), -1.0)
        return len(self.mem_states)

    def save_experience(self):
        """Saves the recent experience to the memory buffer"""
        for state,action,reward,next_state,done in zip(self.mem_states,self.mem_actions,self.mem_rewards,self.mem_next_states,self.mem_done):
            agent_id=self.get_agent_id(state)                
            experience = state, action, reward, next_state, done
            self.agent_dic[agent_id]["memory"].add_experience(*experience)
            self.agent_dic[agent_id]["new_exp_count"]+=1
            # self.agent_dic[agent_id]["has_new_exp"]=True

        # for agent_id in self.agent_dic:
        #     if self.agent_dic[agent_id]["has_new_exp"]
        #         self.agent_dic[agent_id]["new_exp_count"]+=1
        #         self.agent_dic[agent_id]["has_new_exp"]=False

    def take_optimisation_step(self, agents_losses, clipping_norm=None, retain_graph=False):
        """Takes an optimisation step by calculating gradients given the loss and then updating the parameters"""
        # if len(agents_losses)>1:
        #     breakpoint()

        try:
            for (agent_id,loss) in agents_losses:
                self.agent_dic[agent_id]["optimizer"].zero_grad()
            if self.config.does_need_network_state_embeding:
                self.config.GAT_optim.zero_grad()        

        except Exception as e:
            breakpoint()        


        num_losses=len(agents_losses)
        try:
            for idx in range(num_losses):
                loss=agents_losses[idx][1]
                if idx<num_losses-1:
                    loss.backward(retain_graph=self.config.retain_graph)
                else:
                    loss.backward(retain_graph=False)
        except Exception as e:
            breakpoint()

        try:
            if clipping_norm is not None:
                for (agent_id,loss) in agents_losses:
                    torch.nn.utils.clip_grad_norm_(self.agent_dic[agent_id]["policy"].parameters(), clipping_norm) #clip gradients to help stabilise training     
                if self.config.does_need_network_state_embeding:
                    torch.nn.utils.clip_grad_norm_(self.config.GAT_parameters, clipping_norm) #clip gradients to help stabilise training     
                
        except Exception as e:
            breakpoint()

        try:
            for (agent_id,loss) in agents_losses:
                self.agent_dic[agent_id]["optimizer"].step()
            
            if self.config.does_need_network_state_embeding:
                self.config.GAT_optim.step()
        except Exception as e:
           breakpoint()

        # if len(agents_losses)>1:
        #     breakpoint()


        # network=self.agent_dic[agent_id]["policy"]
        # optimizer=self.agent_dic[agent_id]["optimizer"]
        # if not isinstance(network, list): network = [network]
        # optimizer.zero_grad() #reset gradients to 0
        # loss.backward(retain_graph=retain_graph) #this calculates the gradients

        # self.logger.info("Loss -- {}".format(loss.item()))
        # if self.debug_mode: self.log_gradient_and_weight_information(network, optimizer)
        # if clipping_norm is not None:
        #     for net in network:
        #         torch.nn.utils.clip_grad_norm_(net.parameters(), clipping_norm) #clip gradients to help stabilise training
        # optimizer.step() #this applies the gradients
    
    def log_gradient_and_weight_information(self, network, optimizer):

        # log weight information
        total_norm = 0
        for name, param in network.named_parameters():
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        self.logger.info("Gradient Norm {}".format(total_norm))

        for g in optimizer.param_groups:
            learning_rate = g['lr']
            break
        self.logger.info("Learning Rate {}".format(learning_rate))

    def soft_update_of_target_network(self, local_model, target_model, tau):
        """Updates the target network in the direction of the local network but by taking a step size
        less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


    def create_agent_dic(self, key_to_use=None, override_seed=None, hyperparameters=None):
        """Creates a neural network for the agents to use
        env_agent_dic is a dictionary with the road-network lanes as keys and number of actions as values"""
        if hyperparameters is None: hyperparameters = self.hyperparameters
        if key_to_use: hyperparameters = hyperparameters[key_to_use]
        if override_seed: seed = override_seed
        else: seed = self.config.seed

        default_hyperparameter_choices = {"output_activation": None, "hidden_activations": "leakyRelu", "dropout": 0.0,
                                          "initialiser": "default", "batch_norm": False,
                                          "columns_of_data_to_be_embedded": [],
                                          "embedding_dimensions": [], "y_range": ()}

        for key in default_hyperparameter_choices:
            if key not in hyperparameters.keys():
                hyperparameters[key] = default_hyperparameter_choices[key]
        agent_dic={
        agent_id:{\
            "policy":NN(input_dim=self.get_policy_input_size(), 
                    layers_info=hyperparameters["linear_hidden_units"] + [self.get_action_space_size(agent_id)],
                    output_activation=hyperparameters["final_layer_activation"],
                    batch_norm=hyperparameters["batch_norm"], dropout=hyperparameters["dropout"],
                    hidden_activations=hyperparameters["hidden_activations"], initialiser=hyperparameters["initialiser"],
                    columns_of_data_to_be_embedded=hyperparameters["columns_of_data_to_be_embedded"],
                    embedding_dimensions=hyperparameters["embedding_dimensions"], y_range=hyperparameters["y_range"],
                    random_seed=seed).to(self.device),
            "intersec_id_embed_layer":torch.nn.Linear(self.intersection_id_size,self.intersection_id_embedding_size).to(self.device),
            "memory": Replay_Buffer(self.hyperparameters["buffer_size"], self.hyperparameters["batch_size"], self.config.seed, self.device),
            "new_exp_count":0,
            "total_exp_count":0 ,
            }

        for agent_id in self.env_agent_dic
        }

        for agent_id in agent_dic:
            agent_dic[agent_id]["optimizer"]=optim.Adam(list(agent_dic[agent_id]["policy"].parameters())\
                                                        +list(agent_dic[agent_id]["intersec_id_embed_layer"].parameters()),\
                        lr=self.hyperparameters["learning_rate"], eps=1e-4)
               
        return agent_dic    
    
    def get_intersection_id_embedding(self,agent_id,intersec_id,eval=False):
        if eval:
            self.agent_dic[agent_id]["intersec_id_embed_layer"].eval()
            with torch.no_grad():
                output=self.agent_dic[agent_id]["intersec_id_embed_layer"](intersec_id)
            self.agent_dic[agent_id]["intersec_id_embed_layer"].train()
            return self.relu(output)

        return self.relu(self.agent_dic[agent_id]["intersec_id_embed_layer"](intersec_id))

    def get_action_values(self,agent_id,embeding,eval=False):
        if eval:
            self.agent_dic[agent_id]["policy"].eval()
            with torch.no_grad():
                output=self.agent_dic[agent_id]["policy"](embeding)
            self.agent_dic[agent_id]["policy"].train()
            return output
        return self.agent_dic[agent_id]["policy"](embeding)

    def turn_on_any_epsilon_greedy_exploration(self):
        """Turns off all exploration with respect to the epsilon greedy exploration strategy"""
        print("Turning on epsilon greedy exploration")
        self.turn_off_exploration = False

    def turn_off_any_epsilon_greedy_exploration(self):
        """Turns off all exploration with respect to the epsilon greedy exploration strategy"""
        print("Turning off epsilon greedy exploration")
        self.turn_off_exploration = True

    def freeze_all_but_output_layers(self, network):
        """Freezes all layers except the output layer of a network"""
        print("Freezing hidden layers")
        for param in network.named_parameters():
            param_name = param[0]
            assert "hidden" in param_name or "output" in param_name or "embedding" in param_name, "Name {} of network layers not understood".format(param_name)
            if "output" not in param_name:
                param[1].requires_grad = False

    def unfreeze_all_layers(self, network):
        """Unfreezes all layers of a network"""
        print("Unfreezing all layers")
        for param in network.parameters():
            param.requires_grad = True

    @staticmethod
    def move_gradients_one_model_to_another(from_model, to_model, set_from_gradients_to_zero=False):
        """Copies gradients from from_model to to_model"""
        for from_model, to_model in zip(from_model.parameters(), to_model.parameters()):
            to_model._grad = from_model.grad.clone()
            if set_from_gradients_to_zero: from_model._grad = None

    @staticmethod
    def copy_model_over(from_model, to_model):
        """Copies model parameters from from_model to to_model"""
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())

    def get_agent_id(self,state):
        agent_id=state["agent_id"]
        return agent_id


    def time_for_q_network_to_learn(self,agent_id):
        """Returns boolean indicating whether enough steps have been taken for learning to begin and there are
        enough experiences in the replay buffer to learn from"""
        if self.enough_new_exp(agent_id) and self.enough_total_exp(agent_id):
            self.agent_dic[agent_id]["new_exp_count"]=0
            return True
        return False

    def enough_total_exp(self,agent_id):
        """Boolean indicated whether there are enough experiences in the memory buffer to learn from"""
        return len(self.agent_dic[agent_id]["memory"]) > self.hyperparameters["batch_size"]

    def enough_new_exp(self,agent_id):
        """Returns boolean indicating whether enough steps have been taken for learning to begin"""
        return self.agent_dic[agent_id]["new_exp_count"]>=self.hyperparameters["num-new-exp-to-learn"]

    def sample_experiences(self,memory):
        """Draws a random sample of experience from the memory buffer"""
        experiences = memory.sample()
        states, actions, rewards, next_states, dones = experiences
        return states, actions, rewards, next_states, dones

    def save_policies(self):
        """Saves the policy"""
        net_name=self.config.Constants["NETWORK"]
        os.makedirs(f"Models/{self.config.model_version}/{ net_name }/{self.config.routing_mode}", exist_ok=True)
        for agent_id in self.agent_dic:
            torch.save(self.agent_dic[agent_id]["policy"].state_dict(),"Models/{}/{}/{}/agent_{}_policy.pt".format(self.config.model_version,self.config.Constants["NETWORK"],self.config.routing_mode,agent_id))
            torch.save(self.agent_dic[agent_id]["intersec_id_embed_layer"].state_dict(),"Models/{}/{}/{}/agent_{}_id_embed_layer.pt".format(self.config.model_version,self.config.Constants["NETWORK"],self.config.routing_mode,agent_id))

        torch.save(self.config.GAT.state_dict(),"Models/{}/{}/{}/GAT.pt".format(self.config.model_version,self.config.Constants["NETWORK"],self.config.routing_mode))
        
        # torch.save(self.q_network_local.state_dict(), "Models/{}_local_network.pt".format(self.agent_name))

    def load_policies(self):
        for agent_id in self.agent_dic:
            self.agent_dic[agent_id]["policy"].load_state_dict(torch.load("Models/{}/{}/{}/agent_{}_policy.pt".format(self.config.model_version,self.config.Constants["NETWORK"],self.config.routing_mode,agent_id),map_location='cuda:0'))
            self.agent_dic[agent_id]["intersec_id_embed_layer"].load_state_dict(torch.load("Models/{}/{}/{}/agent_{}_id_embed_layer.pt".format(self.config.model_version,self.config.Constants["NETWORK"],self.config.routing_mode,agent_id),map_location='cuda:0'))
        self.config.GAT.load_state_dict(torch.load("Models/{}/{}/{}/GAT.pt".format(self.config.model_version,self.config.Constants["NETWORK"],self.config.routing_mode),map_location='cuda:0'))