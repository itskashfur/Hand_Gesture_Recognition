<h1>Hand-Gesture-Recognition</h1>
<p>This repository stores the source code for the hand gesture recognition model. Please note that you will need python and pip to be installed.</p>
<p>To open the app in streamlit, download <code>requirements.txt</code>, <code>hand_gesture_reader_deployed.py</code>, <code>model_rf__date_time_2023_09_23__12_22_48__acc_1.0__hand__oneimage.pkl</code>.</p>

<p>Next, enter the following command in the command prompt <code>pip install -r requirements.txt</code> to install the necessary libraries.</p>

<p>Open the directory in a terminal where the files are stored and enter the following command <code>streamlit run hand_gesture_reader_deployed.py</code>.</p>

<p>Alternatively, you may open using python by downloading and running the following file: <code>hand_gesture_reader.py</code>. (Make sure to install requirements first.)</p>

<p>When the <code>hand_gesture_reader.py</code> file is run, a webcam window opens which can predict four gestures. The gestures activate the following keys:</p>

<ol>
  <li>Hand Closed - up arrow key</li>
  <li>Hand Three - right arrow key</li>
  <li>Hand Open - left arrow key</li>
  <li>Hand Zero - down arrow key</li>
</ol>

<p>The sample images of gestures can be found at the end of README and in the sample_images folder.</p>

<p>You may change the type of key to be activated by changing the <code>class_to_key</code> dictionary in <code>hand_gesture_reader.py</code> file. Just replace the dictionary values to a string representing the key you wish to activate. For example, you can change the key of the gesture <code>Closed</code> from <code>up</code> to <code>h</code> so that when the 'Hand Closed' gesture is shown, the program will activate the 'H' key. Refer to pyautogui documentation for the available keys.</p>

<p>The <code>hand_gesture_reader.py</code> file uses the random forest model parameters from <code>model_rf__date_time_2023_09_23__12_22_48__acc_1.0__hand__oneimage.pkl</code> file. These parameters were trained in <code>model_hand_rf.py</code> file using the data stored in .npz files. The data was made using <code>hand_landmark_dataset_maker.py</code> file.</p>

<p>Hand Closed <img src="sample_images/Hand Closed.jpg" alt="Hand Closed" /></p>
<p>Hand Three <img src="sample_images/Hand Three.jpg" alt="Hand Three" /></p>
<p>Hand Open <img src="sample_images/Hand Open.jpg" alt="Hand Open" /></p>
<p>Hand Zero <img src="sample_images/Hand Zero.jpg" alt="Hand Zero" /></p>

<!-- <hr /> -->

<!-- <h2>📊 Repository Stats &amp; Views</h2>
<p>To give you an idea of how this project is performing and who is interacting with it, here are the real-time statistics:</p>

<table border="1">
  <thead>
    <tr>
      <th align="left"><strong>Metric</strong></th>
      <th align="left"><strong>Badge / Counter</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left"><strong>Total Profile Views</strong></td>
      <td align="left"><img src="https://komarev.com/ghpvc/?username=itskashfur&amp;color=blue&amp;style=flat-square" alt="Views" /></td>
    </tr>
    <tr>
      <td align="left"><strong>Repository Visitors</strong></td>
      <td align="left"><img src="https://profile-counter.glitch.me/itskashfur/count.svg" alt="Visitor Count" /></td>
    </tr>
    <tr>
      <td align="left"><strong>Repo Stars</strong></td>
      <td align="left"><img src="https://img.shields.io/github/stars/itskashfur/Hand_Gesture_Recognition?style=social" alt="GitHub stars" /></td>
    </tr>
    <tr>
      <td align="left"><strong>Repo Forks</strong></td>
      <td align="left"><img src="https://img.shields.io/github/forks/itskashfur/Hand_Gesture_Recognition?style=social" alt="GitHub forks" /></td>
    </tr>
  </tbody> -->
<!-- </table> -->
