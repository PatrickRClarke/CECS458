# CECS458
A small GUI that makes market predictions using an LSTM hosted on a server.

Everything here can be run in the dl environment detailed in the tutorial video from the beginning of the 
semester with the exception of the files in the Server folder. Those require uvicorn, fastapi, and a valid
Tailscale address for the machine that you would like to connect to. This way the GUI can be used securely
in public while the server runs at home. 
