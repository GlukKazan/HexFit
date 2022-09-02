"use strict";

const readline = require('readline'); 
//const fs = require('fs'); 
const lineByLine = require('n-readlines');

const ml = require('./model');
const game = require('./game');

const SIZE  = 11;
const BATCH = 10; // 100;

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

/*async function onLine(data) {
//  console.log(data);
    await game.proceed(model, SIZE, BATCH, data, logger);
}*/

async function proceed() {
    model = await ml.create(1, SIZE, logger);
    const reader = new lineByLine('data/hex.dat'); 
    let line;
    while (line = reader.next()) { 
        console.log(line.toString()); 
        await game.proceed(model, SIZE, BATCH, line.toString(), logger);
    } 
/*  readline.createInterface({ 
        input: fs.createReadStream('data/hex.dat'), 
        console: false 
    }).on('line', onLine);*/
/*  await game.proceed(model, SIZE, BATCH, 'FaAcFbKcFcBcBbJcJbCcCbIcIbHcFdDcFeGcFfEcGeEiHgGfFgGhDiEhDhEgHhGgHeHfIeIfJeJfDfDgCgChFhFiBhBiAjAiKeKf', logger);
    await game.proceed(model, SIZE, BATCH, 'FaAcFbKcFcBcBbJcJbCcCbIcIbHcFdDcFeGcFfEcGeEiDiEgGgFgGfFiHhGjIiHgGhHiIhHkJjJkIkIjJi', logger);
    await game.proceed(model, SIZE, BATCH, 'CaFfCgChHhHeEgEeDeDfAhBfCfDgBiBhAiBg', logger);
    await game.proceed(model, SIZE, BATCH, 'CaFfCgChHhHeEgEeDeDfAhBfCfDgBiBhBgAi', logger);
    await game.proceed(model, SIZE, BATCH, 'CaFfCgChHhHeEgEeBgDfDgCe', logger);
    await game.proceed(model, SIZE, BATCH, 'CaFfCgChHhHeEgEeBfDfBhCfBgCd', logger);
    await game.proceed(model, SIZE, BATCH, 'CaFfCgChHhHeEgDeEeEfBhBiDgEhDhCjDiDjEiEkEjDkGjFjGhHiIhDfCeCcBdBcCdDcDdEcEdFcGcHbHc', logger);
    await game.proceed(model, SIZE, BATCH, 'CaFfCgChHhHeEgEd', logger);
    await game.proceed(model, SIZE, BATCH, 'CaFfCgChHhHeDgEd', logger);*/
}

function exec() {
    setTimeout(exec, 1000);
}

function run() {
    proceed();
    exec();
}

run();