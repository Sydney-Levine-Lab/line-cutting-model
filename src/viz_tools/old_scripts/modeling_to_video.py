import re
#open this text file which contains the PDDL state history output
with open('/Users/joekwon/Desktop/line_modeling/all_state_histories/recursive/state_history_no_line_1_test.pddl_0.0_iteration1.txt') as f:
	inputtext = f.read()
f.close()

#final output data
EP_DATA = ""
EP_DATA += '{'

#get just the x location parts and y location parts in separate chunks. 
#the output is a list of all these chunks. If the list is 20 long, there were 20 episodes.
xlocation_data = re.findall(r'(?<=:xloc).*?(?=:yloc)', inputtext)
ylocation_data = re.findall(r'(?<=:yloc).*?(?=\)\)\))', inputtext)

#sanity check 
if len(xlocation_data) != len(ylocation_data):
	print("parsing is wack?")

#iterate through all the episodes
for episode in range(len(xlocation_data)):
	#initialize the dictionary for a single episode
	all_agent_locs = [[-1]*2 for _ in range(8)]
	#get the single episode x and y location data
	xdata = xlocation_data[episode]
	ydata = ylocation_data[episode]
	#store data here. will append to the final output data (EP_DATA) at the end
	single_episode_data = ""
	#get the INTEGER VALUE!!! 
	#subtract the value by 1 to adjust it to the indexing differences
	all_agent_locs[0][0] = int(xdata.split('(:agent1,) => ')[1].split(',')[0])-1
	all_agent_locs[0][1] = int(ydata.split('(:agent1,) => ')[1].split(',')[0])-1

	all_agent_locs[1][0] = int(xdata.split('(:agent2,) => ')[1].split(',')[0])-1
	all_agent_locs[1][1] = int(ydata.split('(:agent2,) => ')[1].split(',')[0])-1

	all_agent_locs[2][0] = int(xdata.split('(:agent3,) => ')[1].split(',')[0])-1
	all_agent_locs[2][1] = int(ydata.split('(:agent3,) => ')[1].split(',')[0])-1

	all_agent_locs[3][0] = int(xdata.split('(:agent4,) => ')[1].split(',')[0])-1
	all_agent_locs[3][1] = int(ydata.split('(:agent4,) => ')[1].split(',')[0])-1

	all_agent_locs[4][0] = int(xdata.split('(:agent5,) => ')[1].split(',')[0])-1
	all_agent_locs[4][1] = int(ydata.split('(:agent5,) => ')[1].split(',')[0])-1

	all_agent_locs[5][0] = int(xdata.split('(:agent6,) => ')[1].split(',')[0])-1
	all_agent_locs[5][1] = int(ydata.split('(:agent6,) => ')[1].split(',')[0])-1

	all_agent_locs[6][0] = int(xdata.split('(:agent7,) => ')[1].split(',')[0])-1
	all_agent_locs[6][1] = int(ydata.split('(:agent7,) => ')[1].split(',')[0])-1

	all_agent_locs[7][0] = int(xdata.split('(:agent8,) => ')[1].split(',')[0])-1
	all_agent_locs[7][1] = int(ydata.split('(:agent8,) => ')[1].split(',')[0])-1

	single_episode_data += '{'

	#insert the integers into the format we want
	for agent in range(1,9):
		single_episode_data += str(agent)
		single_episode_data += ':'
		single_episode_data += "{{'coords': ({},{}),'holding': False,'orientation':'D','completed': False}}".format(all_agent_locs[agent-1][1], all_agent_locs[agent-1][0])
		single_episode_data += ','

	single_episode_data += '}'

	EP_DATA += str(episode+1)
	EP_DATA += ':'
	EP_DATA += single_episode_data
	EP_DATA += ','
	
EP_DATA += '}'

with open('/Users/joekwon/Desktop/line_modeling/viz_tools/episode_data_string', 'w') as outfile:
	outfile.write(EP_DATA)
outfile.close()