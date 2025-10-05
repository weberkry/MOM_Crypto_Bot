# What is all this about?

Benoît Mandelbrot’s The Misbehavior of Markets: A Fractal View of Financial Turbulence is a remarkable book that challenges conventional finance and introduces fractal geometry into market analysis. This repository translates some of those ideas into code, with a focus on applying fractal-based methods to estimate risks in cryptocurrencies and the broader stock market.

"I have no financial interest in their success of failure; I am a scientist, not a money man. 
But I wish you godd fortune."  -Mandelbrot


## How to use it

Telegram Bot : CryptoMOM

Use "/start" in the bot to get started  

Use "/help" for information on how to interact with this bot

You can also prompt you question in a different language and the bot will answer accordingly.  
But the translation is not so smooth and needs some attention.


## Notes

This project is still in the very beginning and currently has only the probability density function fitting and hurst calculation available.
The data is stored in an InfluxDB and currently only BTC Data is available, others will follow.

Please be patient with the bot.  
It is running on a RaspberryPi server and may run into performance issues, because depending on the request there will be a massive calculation in the background.  
A response time of about 2-3min is to be expected and if no repsonse is provide please try again later.

