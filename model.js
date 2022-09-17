"use strict";

const _ = require('underscore');
const tf = require('@tensorflow/tfjs-node-gpu');

const BATCH_SIZE  = 256;
const EPOCH_COUNT = 10;
const VALID_SPLIT = 0.1;
const LEARNING_RATE = 0.0001;

const FILE_PREFIX = 'file:///users/valen';

async function init() {
    await tf.ready();
    await tf.enableProdMode();
    console.log(tf.getBackend());
}

async function load(url, logger) {
    const t0 = Date.now();
    await init();
    const model = await tf.loadLayersModel(url);
    const opt = tf.train.sgd(LEARNING_RATE);
    model.compile({optimizer: 'sgd', loss: 'meanSquaredError', metrics: ['accuracy']});
    const t1 = Date.now();
    console.log('Model [' + url + '] loaded: ' + (t1 - t0));
    if (!_.isUndefined(logger)) {
        logger.info('Model [' + url + '] loaded: ' + (t1 - t0));
    }
    return model;
}

async function create(size, logger) {
    const t0 = Date.now();
    await init();

    const input = tf.input({shape: [1, size, size]});
    const z1 = tf.layers.zeroPadding2d({padding: [3, 3], dataFormat: 'channelsFirst'}).apply(input);
    const c1 = tf.layers.conv2d({filters: 64, kernelSize: [7, 7], dataFormat: 'channelsFirst', activation: 'relu'}).apply(z1);

    const z2 = tf.layers.zeroPadding2d({padding: [2, 2], dataFormat: 'channelsFirst'}).apply(c1);
    const c2 = tf.layers.conv2d({filters: 64, kernelSize: [5, 5], dataFormat: 'channelsFirst', activation: 'relu'}).apply(z2);

    const z3 = tf.layers.zeroPadding2d({padding: [2, 2], dataFormat: 'channelsFirst'}).apply(c2);
    const c3 = tf.layers.conv2d({filters: 64, kernelSize: [5, 5], dataFormat: 'channelsFirst', activation: 'relu'}).apply(z3);

    const z4 = tf.layers.zeroPadding2d({padding: [2, 2], dataFormat: 'channelsFirst'}).apply(c3);
    const c4 = tf.layers.conv2d({filters: 48, kernelSize: [5, 5], dataFormat: 'channelsFirst', activation: 'relu'}).apply(z4);

    const z5 = tf.layers.zeroPadding2d({padding: [2, 2], dataFormat: 'channelsFirst'}).apply(c4);
    const c5 = tf.layers.conv2d({filters: 48, kernelSize: [5, 5], dataFormat: 'channelsFirst', activation: 'relu'}).apply(z5);

    const z6 = tf.layers.zeroPadding2d({padding: [2, 2], dataFormat: 'channelsFirst'}).apply(c5);
    const c6 = tf.layers.conv2d({filters: 32, kernelSize: [5, 5], dataFormat: 'channelsFirst', activation: 'relu'}).apply(z6);

    const z7 = tf.layers.zeroPadding2d({padding: [2, 2], dataFormat: 'channelsFirst'}).apply(c6);
    const c7 = tf.layers.conv2d({filters: 32, kernelSize: [5, 5], dataFormat: 'channelsFirst', activation: 'relu'}).apply(z7);

    const fl = tf.layers.flatten().apply(c7);
    const board = tf.layers.dense({units: 512, activation: 'relu'}).apply(fl);

    const action = tf.input({shape: size * size});
    const out = tf.layers.concatenate().apply([board, action]);

    const hidden = tf.layers.dense({units: 512, activation: 'relu'}).apply(out);
    const value = tf.layers.dense({units: 1, activation: 'tanh'}).apply(hidden);

    const model = tf.model({inputs: [input, action], outputs: value});
//  const opt = tf.train.sgd(LEARNING_RATE);
    model.compile({optimizer: 'sgd', loss:'meanSquaredError', metrics: ['accuracy']});

    const t1 = Date.now();
    console.log('Model created: ' + (t1 - t0));
    if (!_.isUndefined(logger)) {
        logger.info('Model created: ' + (t1 - t0));
    }
    return model;
}

async function fit(model, size, x, y, z, batch, logger) {
    const xshape = [batch, 1, size, size];
    const xs = tf.tensor4d(x, xshape, 'float32');
    const yshape = [batch, size * size];
    const ys =  tf.tensor2d(y, yshape, 'float32');
    const zshape = [batch, 1];
    const zs =  tf.tensor2d(z, zshape, 'float32');

    const t0 = Date.now();
    const h = await model.fit([xs, ys], zs, {
        batchSize: BATCH_SIZE,
        epochs: EPOCH_COUNT,
        validationSplit: VALID_SPLIT
    });    

//  console.log(h);
    for (let i = 0; i < EPOCH_COUNT; i++) {
        console.log('epoch = ' + i + ', acc = ' + h.history.acc[i] + ', loss = ' + h.history.loss[i] + ', val_acc = ' + h.history.val_acc[i] + ', val_loss = ' + h.history.val_loss[i]);
        if (!_.isUndefined(logger)) {
            logger.info('epoch = ' + i + ', acc = ' + h.history.acc[i] + ', loss = ' + h.history.loss[i] + ', val_acc = ' + h.history.val_acc[i] + ', val_loss = ' + h.history.val_loss[i]);
        }
    }
    const t1 = Date.now();
    console.log('Fit time: ' + (t1 - t0));
    if (!_.isUndefined(logger)) {
        logger.info('Fit time: ' + (t1 - t0));
    }

    xs.dispose();
    ys.dispose();
    zs.dispose();
}

async function predict(model, size, x, y, batch, logger) {
    const shape = [batch, 1, size, size];
    const xs = tf.tensor4d(x, shape, 'float32');
    const yshape = [batch, size * size];
    const ys =  tf.tensor2d(y, yshape, 'float32');

    const t0 = Date.now();
    const zs = await model.predict([xs, ys]);
    const z = await zs.data();
    const t1 = Date.now();
    console.log('Predict time: ' + (t1 - t0));
    if (!_.isUndefined(logger)) {
        logger.info('Predict time: ' + (t1 - t0));
    }

    xs.dispose();
    ys.dispose();
    zs.dispose();

    return z;
}

async function save(model, fileName) {
    await model.save(`${FILE_PREFIX}/${fileName}`);
}

module.exports.create = create;
module.exports.load = load;
module.exports.fit = fit;
module.exports.predict = predict;
module.exports.save = save;
