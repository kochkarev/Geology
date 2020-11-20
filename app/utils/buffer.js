class BufferHandler {
	
	constructor(processMsgFunction) {
		this.prev = null;
		this.bytesWanted = -1;
		this.processMsgFun = processMsgFunction;
	}

	setBytesWanted(bytesWantToRead) {
		this.bytesWanted = bytesWantToRead;
	}

	processInputBuffer(buf) {
		while (true) {
			let i = buf.indexOf('\r\n');
			if (i === -1) {
				this.prev = !this.prev ? buf : Buffer.concat([this.prev, buf]);
				break;
			} else {
				this.prev = !this.prev ? buf.slice(0, i) : Buffer.concat([this.prev, buf.slice(0, i)]);
				if (this.prev.length < this.bytesWanted) {
					this.prev = Buffer.concat([this.prev, buf.slice(i-1, i+1)]);
					buf = buf.slice(i + 1);
					console.log(`-> ${this.prev.length}, ${this.bytesWanted}, ${buf.length}`);
					continue;
				}
				this.bytesWanted = -1;
				this.processMsgFun(this.prev);
				this.prev = null;
				buf = buf.slice(i + 2);
			}
		}
	}

}

module.exports = {BufferHandler: BufferHandler}