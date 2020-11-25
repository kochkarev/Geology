const { spawn } = require('child_process');
const { BufferHandler } = require('./buffer');


class BackendCommunicator {

	constructor(cfg, handlerArray, handlerSignal, handlerString) {
		let {pyEnvName: pyEnvName, pyEnvPath: pyEnvPath, srcPath: srcPath} = cfg;
		this.bufHandler = new BufferHandler(msg =>  this.processBackendMsg(msg));
		this.hArray = handlerArray;
		this.hSignal = handlerSignal;
		this.hString = handlerString;

		this.state = null;
		this.image_header = null;
		this.proc = spawn(`activate ${pyEnvName} && ${pyEnvPath} ${srcPath}`, {
			shell: true,
		});
		this.proc.stdout.on('data', buf => this.bufHandler.processInputBuffer(buf))
		this.proc.stdout.on('end', () => {console.log('#end')});
		this.proc.stderr.on('data', (data) => console.error(`#err: ${data}`));
		this.proc.on('exit', (code, signal) => console.log(`#exit with code ${code} and signal ${signal}`));
	}

	_sendJSON(json) {
		this.proc.stdin.write(JSON.stringify(json) + '\n');
	}
	
	stop() {
		this._sendJSON({'type': 'shutdown'});
		console.log('#close');
		this.proc.kill();
	}

	ping() {
		this._sendJSON({'type': 'ping'});
	}

	pingImage() {
		this._sendJSON({'type': 'ping_image'});
    }
    
	processBackendMsg(message) {
		if (this.state === null) {
			try {
				let msg_json = JSON.parse(message.toString());
				switch (msg_json.type) {
					case 'string':
						this.hString(msg_json.content);
						break;
					case 'array':
						this.image_header = msg_json;
						this.state = 'array';
						this.bufHandler.setBytesWanted(msg_json.shape.reduce((a, b) => a * b));
						break;
					case 'sig':
						this.hSignal(msg_json.val);
						break;
				}
			} catch (e) {
				console.log(`#unsupported: ${message.toString()}\n`);
			}
		} else if (this.state === 'array') {
			this.hArray(new Uint8Array(message), this.image_header);
			this.state = null;
		}
	}

	
	sendAnnotation(header, data) {
		this._sendJSON(header);
		this.proc.stdin.write(data);
	}

	getFullAnnotation(annotationFullPath, imageId) {
		this._sendJSON({'type': 'get-annotation', 'path': annotationFullPath, 'id': imageId});
	}

}


module.exports = {BackendCommunicator: BackendCommunicator}