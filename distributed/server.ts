
let triangles: number[] = [];
const imageFile = sys.argv[2];
const saveStateFile = sys.argv[1];
const output = "final.png";

type ClientInfo = {
    // Waiting for a result from this client
    pending: boolean,
};

const clientInfo = new Map<WebSocket, ClientInfo>();

function visualize() {
    Deno.run(["--render", "--save-state", "test.state", "-o", output ]);
}

function main() {

    const cppProcess =
}

Deno.serve((req) => {
    const { socket, response } = Deno.upgradeWebSocket(req);

    socket.addEventListener("open", () => {
        console.log("a client connected!");

        clientInfo.set(socket, { pending: false });
    });

    socket.addEventListener("message", (event) => {
        if (event.data === "ping") {
            socket.send("pong");
        }
    });

    socket.addEventListener("close", () => {
        console.log("a client disconnected!");

        clientInfo.delete(socket);
    });

    return response;
});