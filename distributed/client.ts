import type {CppEmittedMessage, ServerEmittedMessage, OptimisationRequest} from "./messages.d.ts";

import process from 'node:process';
import fs from 'node:fs';
import Buffer from 'node:buffer';

// Spawn c++ process --interactive using binary located in argv 1
const cppProcess = Deno.run(process.argv[1], ["--interactive", "--save-state", "test.state", "-i", "ignored", "-o", "ignored" ]);

let triangles: number[] = [];

function loadImage(base64: string) {
    fs.writeFileSync("test.png", Buffer.from(base64, 'base64'));
    cppProcess.stdin.write("i test.png\n");
}

function setTriangles(triangles: number[]) {
    cppProcess.stdin.write("t " + (triangles.length / 10 | 0) + " " + triangles.join(" ") + "\n");
}

function runOptimiser(iterations: number, minTime: number, step: number, norm: OptimisationRequest["norm"]) {
    cppProcess.stdin.write("o " + iterations + " " + minTime + " " + step + " " + norm + "\n");
}

let countIn: number[] = [];

function onCppMessage(message: CppEmittedMessage) {
    const a = countIn.unshift(1);


}

cppProcess.stdout.on('data', (data) => {
    const message = JSON.parse(data.toString());
    onCppMessage(message);
});

function onmessage(message: ServerEmittedMessage) {
    switch (message.type) {
        case "get-image-reply":
            loadImage(message.base64);
            break;
        case "optimisation-request":
            setTriangles(message.triangles);
            runOptimiser(message.iterations, message.minTime, message.step, message.norm);
            break;
    }
}