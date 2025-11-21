import os
import re

input_dir = '/Users/joekwon/Desktop/line_modeling/all_state_histories/naive_recursive_test/'
output_dir = '/Users/joekwon/Desktop/line_modeling/viz_tools/episode_data_strings/'

for filename in os.listdir(input_dir):
    if filename.endswith('_iteration1.txt'):
        input_path = os.path.join(input_dir, filename)
        with open(input_path) as input_file:
            inputtext = input_file.read()
    
            # perform the same processing as before
            EP_DATA = ""
            EP_DATA += '{'

            xlocation_data = re.findall(r'(?<=:xloc).*?(?=:yloc)', inputtext)
            ylocation_data = re.findall(r'(?<=:yloc).*?(?=\)\)\))', inputtext)

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

            # write EP_DATA to file in output_dir
            output_filename = filename.replace('_iteration1.txt', '_episode_data.txt')
            output_path = os.path.join(output_dir, output_filename)
            with open(output_path, 'w') as output_file:
                output_file.write(EP_DATA)
            output_file.close()
        input_file.close()
            
print('Done converting state histories to episode data strings!')


