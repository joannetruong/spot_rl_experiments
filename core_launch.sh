conda activate robostackenv
echo "Killing all tmux sessions..."
tmux kill-server
sleep 2
echo "Starting roscore tmux..."
tmux new -s roscore -d '/home/spot/miniconda3/envs/robostackenv/bin/roscore'
sleep 1
echo "Starting other tmux nodes"
tmux new -s ros_node -d 'cd /home/spot/Research/spot_rl_experiments && /home/spot/miniconda3/envs/robostackenv/bin/python spot_ros_node.py'
sleep 3
tmux ls
