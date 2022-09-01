"use strict";

const _ = require('underscore');
const ml = require('./model');

const LETTERS = 'ABCDEFGHIJKLMN';

let X = null;
let Y = null;
let C = 0;

async function proceed(model, size, batch, data) {
    if (data.length % 2 != 0) return;
    let board = new Float32Array(size * size);
    let winner = (data.length % 4 != 0) ? 1 : -1;
    let player = 1;
    let offset = 0;
    for (let pos = 0; pos < data.length - 1; pos += 2, player = -player) {
        if ((X === null) || (C >= batch)) {
            if (X !== null) {
                await ml.fit(model, size, X, Y, C);
            }
            X = new Float32Array(batch * size * size);
            Y = new Float32Array(batch * size * size);
            C = 0;
        }
        const x = _.indexOf(LETTERS, data[pos]);
        if ((x < 0) || (x >= size)) return;
        const y = _.indexOf(LETTERS, data[pos + 1].toUpperCase());
        if ((y < 0) || (y >= size)) return;
        const move = y * size + x;
        if ((pos > 0) /*&& (winner * player > 0)*/) {
            for (let i = 0; i < size; i++) {
                X[offset + i] = board[i];
            }
            Y[offset + move] = player * winner;
            C++;
            offset += size;
        }
        board[move] = player;
    }
}

module.exports.proceed = proceed;
