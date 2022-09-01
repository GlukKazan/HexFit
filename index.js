"use strict";

const readline = require('readline'); 
const fs = require('fs'); 

const ml = require('./model');
const game = require('./game');

const SIZE  = 11;
const BATCH = 100;

let model = null;
let done = false;

function setDone() {
    done = true;
}

async function onLine(data) {
//  console.log(data);
    await game.proceed(model, SIZE, BATCH, data);
}

async function proceed(callback) {
    model = await ml.create(1, SIZE);
    readline.createInterface({ 
        input: fs.createReadStream('data/hex.dat'), 
        console: false 
    }).on('line', onLine);
    callback();
}

function exec() {
    if (!done) {
        setTimeout(exec, 1000);
    }
}

function run() {
    proceed(setDone);
    exec();
}

run();