"use strict";

const ml = require('./model');
const game = require('./game');

const SIZE = 11;
const URL = 'https://games.dtco.ru/hex-' + SIZE + '/model.json';


async function proceed() {
    const model = await ml.load(URL);
    let X = new Float32Array(SIZE * SIZE);
    let Y = new Float32Array(SIZE * SIZE);
    game.decode('92/92/92/92/92/92/92/92/92/92/92', X, SIZE, 0, 1);
    Y[10] = 1;
    const Z = await ml.predict(model, SIZE, X, Y, 1);
    game.dump(X, SIZE, 0, Y);
    console.log('Value: ' + Z[0]);
}

async function run() {
    await proceed();
}

(async () => { await run(); })();