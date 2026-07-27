declare module '@xhmikosr/decompress' {
  namespace decompress {
    interface File {
      data: Buffer;
      mode: number;
      mtime: string;
      path: string;
      type: string;
    }

    interface DecompressOptions {
      filter?: (file: File) => boolean;
      map?: (file: File) => File;
      plugins?: unknown[];
      strip?: number;
    }
  }

  function decompress(
    input: string | Buffer,
    output?: string,
    options?: decompress.DecompressOptions
  ): Promise<decompress.File[]>;

  export = decompress;
}
