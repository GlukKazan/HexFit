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
    filename: 'gobot-%DATE%.log',
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

async function onLine(data) {
//  console.log(data);
    await game.proceed(model, SIZE, BATCH, data, logger);
}

async function proceed() {
    model = await ml.create(3, SIZE, logger);
    const rl = readline.createInterface({
        input: fs.createReadStream('data/hex-' + SIZE + '.txt'), 
        console: false 
    });
    for await (const line of rl) {
        await game.proceed(model, SIZE, BATCH, line, logger);
    }
    await ml.save(model, 'hex-large-' + SIZE + '.json');
}

async function run() {
    await proceed();
}

(async () => { await run(); })();