"use strict";

const fs = require('fs'); 
const readline = require('readline'); 

const ml = require('./model');
const game = require('./game');

const SIZE  = 11;
const BATCH = 1024;

let model = null;

var winston = require('winston');
require('winston-daily-rotate-file');

const logFormat = winston.format.combine(
    winston.format.timestamp({
        format: 'HH:mm:ss'
    }),
    winston.format.printf(
        info => `${info.level}: ${info.timestamp} - ${info.message}`
    )
);

var transport = new winston.transports.DailyRotateFile({
    dirname: '',
    filename: 'flip-' + SIZE + '-%DATE%.log',
    datePattern: 'YYYY-MM-DD',
    zippedArchive: true,
    maxSize: '20m',
    maxFiles: '14d'
});

var logger = winston.createLogger({
    format: logFormat,
    transports: [
      transport
    ]
});

async function proceed() {
    model = await ml.create(SIZE, logger);
 // model = await ml.load(URL, logger);
    const rl = readline.createInterface({
        input: fs.createReadStream('data/hex-' + SIZE + '.txt'), 
        console: false 
    });
    for await (const line of rl) {
        await game.proceed(model, SIZE, BATCH, line, logger);
    }
    await ml.save(model, 'flip-' + SIZE + '.json');
}

async function run() {
    await proceed();
}

(async () => { await run(); })();