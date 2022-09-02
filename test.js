"use strict";

const ml = require('./model');
const game = require('./game');

const SIZE = 11;

let done = false;

function setDone() {
    done = true;
}

await function proceed(callback) {
    const model = await ml.create(1, SIZE);
    let X = new Float32Array(SIZE * SIZE);
    game.decode('92/2A8/92/92/92/92/92/92/92/8a2/92', X, SIZE, 0, 1);
//  const Y = await ml.predict(model, SIZE, X, 1);
    game.dump(X, SIZE, 0, undefined /*Y*/);
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