import tempfile
import numpy as np
import streamlit as st
import cv2
from PIL import Image
import tempfile
from main import process_video 
from main import get_teams_possession
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cmasher as cmr

# Set the theme for the app
st.set_page_config(
    page_title="Football Players Detection",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
   
)
def create_colors_info(team1_name, team1_player_color, team1_keeper_color, team2_name, team2_player_color, team2_keeper_color):
    """
    Creates and returns a dictionary and list for storing team colors information.

    Parameters:
    - team1_name (str): Name of Team 1.
    - team1_player_color (str): Hexadecimal color code for Team 1 players.
    - team1_keeper_color (str): Hexadecimal color code for Team 1 goalkeeper.
    - team2_name (str): Name of Team 2.
    - team2_player_color (str): Hexadecimal color code for Team 2 players.
    - team2_keeper_color (str): Hexadecimal color code for Team 2 goalkeeper.

    Returns:
    - colors_dic (dict): Dictionary containing team colors information.
    - color_list_lab (list): List of color labels for display or further processing.
    """
    # Initialize an empty dictionary and list
    colors_dic = {}
    color_list_lab = []
    # Convert hexadecimal color codes to RGB values
    team1_player_color_rgb = tuple(int(team1_player_color[i:i+2], 16) for i in (1, 3, 5)) # Convert hex to RGB #FF5733
    team1_keeper_color_rgb = tuple(int(team1_keeper_color[i:i+2], 16) for i in (1, 3, 5))
    team2_player_color_rgb = tuple(int(team2_player_color[i:i+2], 16) for i in (1, 3, 5))
    team2_keeper_color_rgb = tuple(int(team2_keeper_color[i:i+2], 16) for i in (1, 3, 5))

    # Store Team 1 colors information
    colors_dic[team1_name] = [team1_player_color_rgb, team1_keeper_color_rgb]

    # Store Team 2 colors information
    colors_dic[team2_name] = [team2_player_color_rgb, team2_keeper_color_rgb]

    # Add color labels to the list
    color_list_lab.append(f"{team1_name} Player Color")
    color_list_lab.append(f"{team1_name} Goalkeeper Color")
    color_list_lab.append(f"{team2_name} Player Color")
    color_list_lab.append(f"{team2_name} Goalkeeper Color")

    return colors_dic


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def get_team_name(player_id, dataframe):
    """
    Get the team name for a given player ID.
    
    Parameters:
    - player_id: The ID of the player.
    - dataframe: The DataFrame containing the player data.
    
    Returns:
    - team_name: The name of the team the player belongs to.
    """
    # Query the DataFrame for the given player ID
    player_row = dataframe[dataframe['player_ID'] == player_id]
    
    # Check if the player exists in the DataFrame
    if not player_row.empty:
        # Return the team name
        return player_row['player_team'].values[0]
    else:
        # If player ID is not found
        return None
    
def plot_passes(pass_data, team , player_data, pitch_image='pitch.jpg'):
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    # Display the football pitch image
    print(team)
    ax.imshow(pitch_image, extent=[0, 1920, 0, 1080])
    color_point = team1_color if team == team1_name else team2_color
    print(color_point)
    # Plot current pass
    for from_p, to_p in zip(pass_data['from'], pass_data['to']):
    # for index, (from_p, to_p) in enumerate(zip(pass_data['from'], pass_data['to'])):
        if team == get_team_name(from_p, player_data) and team == get_team_name(to_p, player_data):
            start_pos = (pass_data['start_pos_X'], 1080 - pass_data['start_pos_Y'])  # Mirror the y-coordinate
            end_pos = (pass_data['end_pos_X'], 1080 - pass_data['end_pos_Y'])  # Mirror the y-coordinate
            
            # Plot the pass line
            ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], color='yellow', linewidth=2)
            ax.scatter([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], c= [color_point], s=30)

    # Customize plot
    # ax.set_xlim(0, 1920)
    # ax.set_ylim(0, 1080)
    fig.patch.set_facecolor('none')
    # ax.set_title(f'Pass Network - Pass {current_pass_index + 1}/{len(pass_data)}')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('on')

    return fig

# def heatmap(df, team, max_frame):
#     kde = sns.kdeplot(
#         data=df[(df.frame_nbr>=1) & (df.frame_nbr<=max_frame) & (df.player_team == team)], 
#         x='player_pos_X', 
#         y='player_pos_Y',
#         fill=True,
#         thresh=.05,
#         alpha=.5,
#         n_levels=20,
#         cmap=cmr.ember,
#         zorder=1
#     )

    
#     return kde
def heatmap(df, team, max_frame, ax):
    kde = sns.kdeplot(
        data=df[(df.frame_nbr>=1) & (df.frame_nbr<=max_frame) & (df.player_team == team)], 
        x='player_pos_X', 
        y='player_pos_Y',
        fill=True,
        thresh=.05,
        alpha=.5,
        n_levels=20,
        cmap=cmr.ember,
        zorder=1,
        ax=ax
    )
    return kde
# def plot_players_pos(player_data, colors_dic, pitch_image,team, player_index=None):
#     fig, ax = plt.subplots(figsize=(10, 7))

#     # Display the football pitch image
#     ax.imshow(pitch_image, extent=[0, 1920, 0, 1080])
#     color = colors_dic.get(team, [(0, 0, 255)])  # Use a default color if the team is not in the dictionary
#     color = [tuple([x/255 for x in c]) for c in color]  # Normalize the RGB values
#     # Plot player positions
#     if player_index is not None:
#         # team = player_data[player_data['player_ID']==player_index]['player_team'].unique()[0]
#         player =  player_data[player_data['player_ID'] == player_index]
#         player_pos = (player['player_pos_X'], 1080 - player['player_pos_Y'])
#         ax.scatter(player_pos[0], player_pos[1], color=color[0], s=20, label=f'Player {player_index}')
        
#     else:
#         # teams = player_data['player_team'].unique()
#         # for team in teams:
#         team_data = player_data[player_data['player_team'] == team]
#         ax.scatter(team_data['player_pos_X'], 1080-team_data['player_pos_Y'], color=color[0], s=20, label=f'{team} Players')

#     fig.patch.set_facecolor('none')
#     # Customize plot
#     ax.legend()
#     # ax.set_xlim(0, 1920)
#     # ax.set_ylim(0, 1080)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     # ax.background_patch.set_facecolor('none')
#     ax.patch.set_facecolor('none')

#     return fig

def plot_players_pos(player_data, colors_dic, team ,pitch_image):
    fig, ax = plt.subplots(figsize=(10, 7))

    # Display the football pitch image
    ax.imshow(pitch_image, extent=[0, 1920, 0, 1080])

    # Plot player positions
    # teams = player_data['player_team'].unique()
    # for team in team:
    team_data = player_data[player_data['player_team'] == team]
    color = colors_dic.get(team, [(0, 0, 255)])  # Use a default color if the team is not in the dictionary
    color = [tuple([x/255 for x in c]) for c in color]  # Normalize the RGB values
    ax.scatter(team_data['player_pos_X'], 1080-team_data['player_pos_Y'], color=color[0], s=20, label=f'{team} Players')

    # Plot current pass

    # Customize plot
    ax.legend()
    # ax.set_xlim(0, 1920)
    # ax.set_ylim(0, 1080)
    fig.patch.set_facecolor('none')
    ax.set_xticks([])
    ax.set_yticks([])

    return fig

def possession_pie(data, colors_dic):
    # Normalize colors to be between 0 and 1
    colors_dic = {team: [(r/255, g/255, b/255) for r, g, b in colors] for team, colors in colors_dic.items()}
    keys = list(data.keys())[:2]
    values = [data[key] for key in keys]

    # Define colors and explode settings for the pie chart
    colors = [colors_dic[key][0] for key in keys]

    explode = (0.1, 0)  # explode the first slice

    # # Create a styled pie chart
    # plt.figure(figsize=(5, 5))
    # plt.pie(values, explode=explode, labels=keys, colors=colors, autopct='%1.1f%%', shadow=True, startangle=100)
    fig, ax = plt.subplots()
    ax.pie(values, explode=explode, labels=keys, colors=colors, autopct='%1.1f%%', shadow=True, startangle=100,  textprops={'color':"white"})
    fig.patch.set_facecolor('none')  # Set the background color to non
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    # Display the pie chart in Streamlit
    st.pyplot(fig)

def main():
     # Display How to Use guide on the main page
    
    
    path_vid = 'input/'
    st.sidebar.title("Football Match Analyzer")
    # demo_selected = st.sidebar.radio(label="Select Demo Video", options=["Demo 1", "Demo 2"], horizontal=True)

    ## Sidebar Setup
    st.sidebar.markdown('---')
    st.sidebar.subheader("Video Upload")
    input_vide_file = st.sidebar.file_uploader('Upload a video file', type=['mp4','mov', 'avi', 'm4v', 'asf'])
    st.sidebar.markdown('---')
    video_path = os.path.join(path_vid, input_vide_file.name) if input_vide_file else None
    
    
    # Tabs for different functionalities 
    tab1, tab2 , tab3= st.tabs(["How to use?", "Process Video", "Visuals and Metrics"])    
    # Tab 1: How to use?
    with tab1:
        st.title("Football Players Detection Web Application")
        st.markdown("""
        ## How to Use This Application

        ### Step 1: Upload the Video
        1. **Upload a Video File**:
        - In the sidebar, go to the "Video Upload" section.
        - Click on the "Upload a video file" button and select your video file (supported formats: mp4, mov, avi, m4v, asf).
        - The selected video file will be displayed in the sidebar.

        ### Step 2: Configure Team Settings
        1. **Access the Team Colors Tab**:
        - Go to the "Team Colors" tab to configure the settings for the two teams playing in the match.

        2. **Enter Team Names and Colors**:
        - **Team 1**:
            - Enter the name of Team 1 in the "Enter Team 1 Name" field.
            - Select the player color for Team 1 using the color picker labeled "Select [Team 1 Name] Player Color".
            - Select the goalkeeper color for Team 1 using the color picker labeled "Select [Team 1 Name] Goalkeeper Color".
        - **Team 2**:
            - Enter the name of Team 2 in the "Enter Team 2 Name" field.
            - Select the player color for Team 2 using the color picker labeled "Select [Team 2 Name] Player Color".
            - Select the goalkeeper color for Team 2 using the color picker labeled "Select [Team 2 Name] Goalkeeper Color".

        3. **Display Video Frame**:
        - Once the video file is uploaded, the first frame of the video will be displayed in the "Video frame" section.
        - Ensure the displayed frame is appropriate for color picking.

        ### Step 3: Start the Detection Process
        1. **Start Detection**:
        - Ensure that you have provided the team names and uploaded the video file.
        - Click on the "Start Detection" button to initiate the detection process.
        - The app will process the video, detect players, referees, and the ball, assign teams based on the colors provided, and generate visualizations.

        ### Step 4: Monitor the Detection Process
        1. **Real-Time Updates**:
        - The progress and status of the detection process will be displayed.
        - The application will toast messages such as "Detection Started!" and "Detection Completed!" based on the progress.

        ### Step 5: View and Analyze Results
        1. **Visualization and Metrics**:
        - Once the detection process is complete, the annotated video frames with detected players, referees, and the ball will be displayed.
        - Team assignments and player positions will be visualized on the tactical map.

        2. **Stop Detection**:
        - If needed, you can stop the detection process by clicking on the "Stop Detection" button.
        """)

    # Tab 2: Team Colors
    with tab2:
        st.title('Team Settings')
        # Video display section
        st.title('Video frame')

        # Upload a video file
        # uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi"])

        if input_vide_file is not None:
            # Temporary file path for the uploaded video
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(input_vide_file.read())
            temp_file.close()

            # OpenCV video capture
            cap = cv2.VideoCapture(video_path)

            # Read the first frame only
            ret, frame = cap.read()

            if ret:
                # Display the first video frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(frame, channels="RGB", use_column_width=True)

            cap.release()

            # # Remove temporary file
            # os.unlink(temp_file.name)


        # Input fields for team names and colors
        global team1_name, team1_color, team1_gk_color, team2_name, team2_color, team2_gk_color
        team1_name = st.text_input("Enter Team 1 Name", "Team 1")
        team1_color = st.color_picker(f"Select {team1_name} Player Color", "#FF5733")
        team1_gk_color = st.color_picker(f"Select {team1_name} Goalkeeper Color",  "#FF5733")

        team2_name = st.text_input("Enter Team 2 Name", "Team 2")
        team2_color = st.color_picker(f"Select {team2_name} Player Color", "#334CFF")
        team2_gk_color = st.color_picker(f"Select {team2_name} Goalkeeper Color", "#334CFF")
        colors_dic = create_colors_info(team1_name, team1_color, team1_gk_color, team2_name, team2_color, team2_gk_color)
        # Ensure session state initialization
        if f"{team1_name} P color" not in st.session_state:
            st.session_state[f"{team1_name} P color"] = "#FFFFFF"  # Replace with default color
        if f"{team1_name} GK color" not in st.session_state:
            st.session_state[f"{team1_name} GK color"] = team1_gk_color  # Initialize with selected GK color
        if f"{team2_name} P color" not in st.session_state:
            st.session_state[f"{team2_name} P color"] = "#FFFFFF"  # Replace with default color
        if f"{team2_name} GK color" not in st.session_state:
            st.session_state[f"{team2_name} GK color"] = team2_gk_color  # Initialize with selected GK color

        stcol1, stcol2 = st.columns([1,1])
        with stcol1:
            ready = not (team1_name and team2_name and input_vide_file)
            start_detection = st.button('Start Detection', disabled=False)
        with stcol2:   
            stop_btn_state = not start_detection
            stop_detection = st.button('Stop Detection', disabled=stop_btn_state)

    # Tab 3: Model Hyperparameters & Detection
    with tab3:
        st.title('Visuals and Metrics')
        st.subheader("Output Video")
        video_file = open('input/output.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)
        
        # Load CSV files
        ball_data = pd.read_csv('ball_data.csv')
        pass_data = pd.read_csv('pass_data.csv')
        player_data = pd.read_csv('player_data.csv')
        st.subheader("Viusalization settings")
        team = st.selectbox("Select Team", player_data['player_team'].unique(), key='team_selectbox')

        # Load football pitch image
        pitch_image = Image.open('pitch.jpg')
        subtab1, subtab2 = st.columns([1,1])
        with subtab1:
            
            st.subheader("Pass Network")
            # Session state to keep track of current pass index
            # if 'current_pass_index' not in st.session_state:
            #     st.session_state.current_pass_index = 0
            npcol1, npcol2 = st.columns([1,1])
            # with npcol1:
            #     if st.button('Previous Pass'):
            #         st.session_state.current_pass_index = max(0, st.session_state.current_pass_index - 1)
            # with npcol2:   
            #     if st.button('Next Pass'):
            #         st.session_state.current_pass_index = min(len(pass_data) - 1, st.session_state.current_pass_index + 1)

            # Plot the data on the pitch
            fig = plot_passes(pass_data, team,player_data, pitch_image)
            st.pyplot(fig)
            
            # st.tab3.markdown('---')
            
            st.subheader("Players positions")
            fig = plot_players_pos(player_data  , colors_dic,team, pitch_image)
            st.pyplot(fig)
            
           
            data = {
                'Chelsea': 60,
                'Man City': 40
            }
            
        with subtab2:    
            st.subheader("Heatmap of Players Positions")
            
            # heatmap_team = st.selectbox("Select Team", player_data['player_team'].unique(), key='heatmap_team_selectbox')
            heatmap_max_frame = st.text_input("Enter the period of time in minutes", "1")
            fig, ax = plt.subplots(figsize=(12.8, 7.2))
            fig.patch.set_facecolor('none')
            # Display the football pitch image
            ax.imshow(pitch_image, extent=[0, 1920, 0, 1080])
            ax.legend()
            # ax.set_xlim(0, 1920)
            # ax.set_ylim(0, 1080)
            # ax.set_title(f'Heatmap of Players Positions for {heatmap_team}')
            ax.set_xticks([])
            ax.set_yticks([])
            st.pyplot(heatmap(player_data, team, int(heatmap_max_frame)*30,ax).get_figure(),clear_figure=True)
            teams = list(colors_dic.keys())
            # st.write(colors_dic)
            
            possession = get_teams_possession(player_data, teams,500)
            st.subheader("Possession")
            possession_pie(data, colors_dic)
        # st.write(possession)
        # st.write(list(colors_dic.values())[0])
        # st.altair_chart(possession)
        
        
    stframe = st.empty()
    # cap = cv2.VideoCapture(tempf.name) if input_vide_file else None
    status = False
    # Detection process
    if start_detection and not stop_detection:
        st.toast('Detection Started!')
        status = process_video(video_path, colors_dic )
    else:
        try:
            cap.release()
        except:
            pass
    if status:
        st.toast('Detection Completed!')
        cap.release()

if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
