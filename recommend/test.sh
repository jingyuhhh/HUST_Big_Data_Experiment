
#!/bin/bash

user_id=629

#python ./recommend.py --level "simple" --mode "content" --user_id $user_id &
#
#python ./recommend.py --level "enhanced" --mode "content" --user_id $user_id &

python ./recommend.py --level "enhanced" --mode "collaborative" --user_id $user_id &


python ./recommend.py --level "simple" --mode "collaborative" --user_id $user_id &

wait





