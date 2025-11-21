import os
import re

input_dir = '/Users/joekwon/Desktop/line_modeling/all_state_histories/recursive/'
output_dir = '/Users/joekwon/Desktop/line_modeling/viz_tools/episode_data_strings/'

for filename in os.listdir(input_dir):
    if not filename.endswith('_iteration1.txt'):
        continue

    input_path = os.path.join(input_dir, filename)

    with open(input_path) as input_file:
        input_text = input_file.read()

        x_location_data = re.findall(r'(?<=:xloc).*?(?=:yloc)', input_text)
        y_location_data = re.findall(r'(?<=:yloc).*?(?=\)\)\))', input_text)

        if len(x_location_data) != len(y_location_data):
            print("Parsing is incorrect.")

        episode_data = {}
        for episode in range(len(x_location_data)):
            all_agent_locs = [[-1]*2 for _ in range(8)]
            x_data = x_location_data[episode]
            y_data = y_location_data[episode]
            single_episode_data = {}

            for agent in range(1, 9):
                agent_str = f':agent{agent},'
                x_coord = int(x_data.split(f'({agent_str}) => ')[1].split(',')[0]) - 1
                y_coord = int(y_data.split(f'({agent_str}) => ')[1].split(',')[0]) - 1
                single_episode_data[str(agent)] = {
                    'coords': (y_coord, x_coord),
                    'holding': False,
                    'orientation': 'D',
                    'completed': False
                }

            episode_data[str(episode + 1)] = single_episode_data

        output_filename = filename.replace('_iteration1.txt', '_episode_data.txt')
        output_path = os.path.join(output_dir, output_filename)

        with open(output_path, 'w') as output_file:
            output_file.write(str(episode_data))

    print(f"Done converting {input_path} to episode data string.")

print('Done converting state histories to episode data strings!')