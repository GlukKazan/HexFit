"use strict";

const ml = require('./model');
const game = require('./game');

const URL = 'https://games.dtco.ru/hex-11/model.json';

const SIZE = 11;

async function proceed() {
    const model = await ml.load(URL);
    let X = new Float32Array(SIZE * SIZE);
    game.decode('92/92/A91/92/7a3/5aB3/7A3/7a3/4b5/6A4/92', X, SIZE, 0, 1);
    const Y = await ml.predict(model, SIZE, X, 1);
    game.dump(X, SIZE, 0, Y);
}

async function run() {
    await proceed();
}

(async () => { await run(); })();