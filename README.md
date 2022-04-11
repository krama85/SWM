Run trained world models on observations and actions to obtain state representations. Trained models are in models_128 folder. 

Example : python run_model.py --save-folder /home/kandan/swm/models_128/UnObserved/AE_0/AE_NLL_medium_8_Sets runs the trained AE (autoencoder) model. 
The state representation of current observation is stored in state and after the action is stored in next_state. 