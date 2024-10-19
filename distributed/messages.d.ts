

type GetStateRequest = {
    type: "get-image-request",
};

type GetImageReply = {
    type: "get-image-reply",
    base64: string,
    triangles: number[],
};

export type OptimisationRequest = {
    type: "optimisation-request",
    triangles: number[],
    iterations: number,
    minTime: number,
    step: number,
    norm: "l1" | "l2" | "l2_squared" | "redmean",
};

type OptimisationReply = {
    bestTriangle: [number, number, number],
    bestColour: [number, number, number, number],
    improvement: number,
    triangleCount: number,
};

export type ClientEmittedMessage = GetStateRequest | OptimisationReply;
export type ServerEmittedMessage = GetImageReply | OptimisationRequest;

export type CppEmittedMessage = {
    type: "triangle",
    p0: [number, number],
    p1: [number, number],
    p2: [number, number],
    color: [number, number, number],
    alpha: number,
    improvement: number,
};