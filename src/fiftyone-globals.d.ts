// Ambient types for FiftyOne's runtime globals; not published to npm.
declare module "@fiftyone/plugins" {
  export enum PluginComponentType {
    Component = "Component",
  }
  export function registerComponent(options: {
    name: string;
    label?: string;
    component: any;
    type: PluginComponentType;
    activator?: (ctx: any) => boolean;
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
}
