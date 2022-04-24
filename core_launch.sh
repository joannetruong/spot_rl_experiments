conda activate outdoor-nav
echo "Killing all tmux sessions..."
tmux kill-server
sleep 2
echo "Starting roscore tmux..."
tmux new -s roscore -d '/home/spot/anaconda3/envs/outdoor-nav/bin/roscore'
sleep 1
echo "Starting other tmux nodes"
tmux new -s ros_node -d 'cd /home/spot/repos/outdoor_nav/spot_rl_experiments && /home/spot/anaconda3/envs/outdoor-nav/bin/python spot_ros_node.py'
sleep 3
tmux ls
