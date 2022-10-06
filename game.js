"use strict";

const _ = require('underscore');
const ml = require('./model');

const LETTERS = 'ABCDEFGHIJKLMNabcdefghijklmn';

let X = null;
let Y = null;
let Z = null;

let C = 0;
let xo = 0;
let yo = 0;

let cnt = 0;

function dump(board, size, offset, moves) {
    for (let y = 0; y < size; y++) {
        let s = '';
        for (let i = 0; i <= y; i++) {
            s = s + ' ';
        }
        for (let x = 0; x < size; x++) {
            const pos = y * size + x;
            if (board[offset + pos] > 0) {
                s = s + '* ';
            } else if (board[offset + pos] < 0) {
                s = s + 'o ';
            }  else if (!_.isUndefined(moves) && (moves[offset + pos] > 1 / (size * size))) {
                s = s + '+ ';
            }  else if (!_.isUndefined(moves) && (moves[offset + pos] < -1 / (size * size))) {
                s = s + 'X ';
            }  else {
                s = s + '. ';
            }
        }
        console.log(s);
    }
    console.log('');
}

function rotate(pos, size, ix) {
    if (ix == 0) return pos;
    const x = pos % size;
    const y = (pos / size) | 0;
    return ((size - 1) - y) * size + (size - 1) - x;
}

function encode(board, size, player, offset, X, ix) {
    if (ml.PLANE_COUNT == 1) {
        for (let pos = 0; pos < size * size; pos++) {
            X[offset + rotate(pos, size, ix)] = board[pos] * player;
        }
    } else {
        const po = size * size;
        for (let pos = 0; pos < size * size; pos++) {
            if (board[pos] * player > 0.01) {
                X[offset + rotate(pos, size, ix)] = 1;
            }
            if (board[pos] * player < -0.01) {
                X[offset + po + rotate(pos, size, ix)] = 1;
            }
        }
    }
}

function isDigit(c) {
    if (c == '-') return true;
    return (c >= '0') && (c <= '9');
}

async function proceed(model, size, batch, data, logger) {
    if (data.length % 2 != 0) return;
    let board = new Float32Array(size * size);
    let winner = (data.length % 4 != 0) ? 1 : -1;
    let player = 1;
    let pos = 0;
    while (pos < data.length - 1) {
        let estimate = 0; let s = 0.1;
        while ((pos < data.length) && isDigit(data[pos])) {
            if (data[pos] == '-') {
                s = -s;
                continue;
            }
            estimate += +data[pos] * s;
            s = s / 10;
        }
        const x = _.indexOf(LETTERS, data[pos]);
        if ((x < 0) || (x >= size)) return;
        const y = _.indexOf(LETTERS, data[pos + 1].toUpperCase());
        if ((y < 0) || (y >= size)) return;
        pos += 2;
        const move = y * size + x;
        for (let ix = 0; ix < 2; ix++) {
            if ((X === null) || (C >= batch)) {
                if (X !== null) {
                    await ml.fit(model, size, X, Y, Z, C, logger);
                    cnt++;
                    if ((cnt % 1000) == 0) {
                        await ml.save(model, 'adagrad-' + ml.PLANE_COUNT + '-' + size + '-' + cnt + '.json');
                        console.log('Save [' + cnt + ']: ' + data);
                        logger.info('Save [' + cnt + ']: ' + data);
                    }
                }
                xo = 0; yo = 0;
                X = new Float32Array(ml.PLANE_COUNT * batch * size * size);
                Y = new Float32Array(batch * size * size);
                Z = new Float32Array(batch);
                C = 0;
            }
            encode(board, size, player, xo, X, ix);
            const r = (winner - estimate) * player;
            if (r > 0) {
                Y[yo + rotate(move, size, ix)] = r;
            }
            Z[C] = winner * player;
//          dump(X, size, offset, Y);
            xo += size * size * ml.PLANE_COUNT;
            yo += size * size;
            C++;
        }
        board[move] = player;
        player = -player;
    }
}

function decode(fen, board, size, offset, player) {
    let pos = 0;
    for (let i = 0; i < fen.length; i++) {
        const c = fen[i];
        if (c != '/') {
            if ((c >= '0') && (c <= '9')) {
                pos += +c;
            } else {
                let ix = _.indexOf(LETTERS, c);
                if (ix >= 0) {
                    let p = 1;
                    if (ix >= 14) {
                        p = -p;
                        ix -= 14;
                    }
                    ix++;
                    for (; ix > 0; ix--) {
                        board[offset + pos] = p * player;
                        pos++;
                    }
                }
            }
            if (pos >= size * size) break;
        } 
    }
}

module.exports.dump = dump;
module.exports.proceed = proceed;
module.exports.decode = decode;
