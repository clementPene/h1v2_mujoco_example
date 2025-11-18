here is an exemple on how to simulate H1V2 in mujoco, and visualize some cyclonedds topics.

Exemple is build following others exemples created by Unitree, using 2 repository :
- [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)
- [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python)

[cyclonedds](https://github.com/eclipse-cyclonedds/cyclonedds) has to be installed.



### UTILISATION

```bash
git clone git@github.com:clementPene/h1v2_mujoco_example.git

cd h1v2_mujoco_example

vcs import . < dependencies/unitree.repos
```

You can then launch a first exemple :
```bash
python3 h1v2_mujoco.py
```

From this point, you will be able to see cyclonedds topics.
To do so, you need to configure cyclonedds. An example of config file is provided :
```bash 
# configure your bashrc with install path
export CYCLONEDDS_HOME="install_path"
export PATH=$CYCLONEDDS_HOME/bin:$PATH

# source each bash terminal
export CYCLONEDDS_URI=file://$PWD/cyclonedds.xml

# visualize topics 
# list active topics
cyclonedds ls --id 79

# subscribe to a topic
cyclonedds typeof rt/lowstate --id 79
```


A controller example is also provided.
```bash
python3 test_controller.py
```
