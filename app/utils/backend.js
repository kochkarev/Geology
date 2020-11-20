const { spawn } = require('child_process');
const { BufferHandler } = require('./buffer');


class BackendCommunicator {

	constructor(cfg, handler_array, handler_signal, handler_string) {
		let {pyenv_name, pyenv_path, src_path} = cfg;
		this.bufferHandler = new BufferHandler(msg =>  this.process_backend_message(msg));
		this.handler_array = handler_array;
		this.handler_signal = handler_signal;
		this.handler_string = handler_string;

		this.state = null;
		this.image_header = null;
		this.proc = spawn(`activate ${pyenv_name} && ${pyenv_path} ${src_path}`, {
			shell: true,
		});
		this.proc.stdout.on('data', buf => this.bufferHandler.processInputBuffer(buf))
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

	ping_image() {
		this._sendJSON({'type': 'ping_image'});
    }
    
	process_backend_message(message) {
		if (this.state === null) {
			try {
				let msg_json = JSON.parse(message.toString());
				switch (msg_json.type) {
					case 'string':
						this.handler_string(msg_json.content);
						break;
					case 'array':
						this.image_header = msg_json;
						this.state = 'array';
						this.bufferHandler.setBytesWanted(msg_json.shape.reduce((a, b) => a * b));
						break;
					case 'sig':
						this.handler_signal(msg_json.val);
						break;
				}
			} catch (e) {
				console.log(`#unsupported: ${message.toString()}\n`);
			}
		} else if (this.state === 'array') {
			this.handler_array(new Uint8Array(message), this.image_header);
			this.state = null;
		}
	}

	
	send_annotation(header, data) {
		this._sendJSON(header);
		this.proc.stdin.write(data);
	}

	getFullAnnotation(annotationFullPath, imageId) {
		this._sendJSON({'type': 'get-annotation', 'path': annotationFullPath, 'id': imageId});
	}

}


module.exports = {BackendCommunicator: BackendCommunicator}