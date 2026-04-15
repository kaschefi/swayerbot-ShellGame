**tracking the right cup**
the challenge is that when a cup is behind another one and in 2d it's hard to 
know which one is the right one.
thats the steps that we are taking to solve this challenge.

**1: Environment & Calibration**
we will use a laptop to simulate the environment.
we need to simulate Sawyer’s perspective.

Homography Setup: Tape four small dots on the table to form a rectangle. This is the "Workspace."

The Code Task: Write a script where you click these four dots on the screen, and OpenCV generates the Bird’s Eye View.
we want to see the cups from the top, even if the camera is at the side.

**2: Detection**
we need to detect two things: The Ball and the Cups.

The Ball: Since the ball is only visible at the start and end, use a simple Color Mask (HSV thresholding). 
It’s faster than a neural network and very reliable for a single bright color.

The Cups: Use a lightweight YOLOv10 model.

we probably don't need to train it on "cups" specifically if we use a pre-trained model;
"bowl" or "cup" is usually already in the COCO dataset.

The Code Task: Create a script that draws bounding boxes around all 3 cups and the ball simultaneously.

**3: The "Association" Logic**
we need to tag the cup that has the ball.

Initial Check: Before the shuffle starts, the code checks which Cup Bounding Box contains 
the Ball Bounding Box.

The Flag: Assign a boolean to that specific Cup ID: is_winning_cup = True.

**4: Tracking**
we need to maintain the "Winning Cup" ID even when it moves and overlaps with others.

Implementation: Use the SORT (Simple Online and Realtime Tracking) algorithm. It’s a tiny Python file that combines a Kalman Filter (to predict motion) and the Hungarian Algorithm (to match IDs).

The Code Task: During the shuffle, the IDs (0, 1, 2) must stay attached to their specific cups. Even if you swap Cup 0 and Cup 1, the number "0" must follow its cup.

**5: The "State Machine"**
the Python script needs to know what "phase" of the game it is in. 
we can control this with the keyboard for now:

Press 'S' (Start): The robot/computer identifies where the ball is and "locks" the ID.

Shuffle Phase: The tracker runs. The ball is invisible, but the "Winning ID" is tracked.

Press 'E' (End): The shuffle stops. The code highlights the winning cup on the screen with a green box.