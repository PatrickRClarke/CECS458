# CECS458
A small GUI that makes market predictions using an LSTM hosted on a server.

Everything here can be run in the "dl" environment detailed in the tutorial video from the beginning of the 
semester with the exception of the files in the Server folder. Those require uvicorn, fastapi, and a valid
Tailscale address for the machine that you would like to connect to. This way the server can be accessed by 
a GUI securely on a VPN, using public WiFi, while the server runs at home. 

**Important**:
While predicting the Estimated Moving Average (EMA), within a window, for the next 5 minute candle may be 
beneficial for some traders - I would not recommend trading by this metric alone. The predictions made by 
this model alone can not be relied upon for consistent profits. I would not recommend relying on this model 
for execution logic. However, it still has value in that it could help a trader understand unfolding trends.

