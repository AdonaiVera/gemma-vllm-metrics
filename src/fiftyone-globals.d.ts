// @fiftyone/plugins and @fiftyone/operators are not published to npm -- they
// only exist as globals injected by the running FiftyOne App at runtime (see
// the `external`/`globals` mapping in vite.config.ts). These ambient
// declarations just keep the TypeScript compiler happy about the imports;
// the real types live in the FiftyOne App itself.
declare module "@fiftyone/plugins" {
  export enum PluginComponentType {
    Panel = "Panel",
    Component = "Component",
  }
  export function registerComponent(options: {
    name: string;
    label?: string;
    component: any;
    type: PluginComponentType;
    activator?: (ctx: any) => boolean;
    surfaces?: string;
    panelOptions?: { surfaces: string };
  }): void;
}

declare module "@fiftyone/operators" {
  export function usePanelEvent(): (
    method: string,
    options: {
      operator: string;
      params: Record<string, any>;
      callback: (result: any) => void;
    }
  ) => void;
  export function useOperatorExecutor(uri: string): {
    execute: (params: Record<string, any>) => void;
    result: any;
    error: any;
    isLoading: boolean;
  };
}
