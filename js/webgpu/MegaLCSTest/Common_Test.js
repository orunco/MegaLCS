function ArrayEqual_ForTest(a, b) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
        if (a[i] !== b[i]) return false;
    }
    return true;
}

function assert(condition, message) {
    if (!condition) {
        throw new Error("Assertion failed: " + message);
    }
}