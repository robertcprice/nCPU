//! HTTP/Web Stack for nCPU/nSynth
//!
//! Complete HTTP implementation for synthesized web applications.

pub mod advanced;
pub mod advanced_web;
pub mod client;
pub mod cookies;
pub mod css;
pub mod event_loop;
pub mod forms;
pub mod fullstack_frameworks;
pub mod graphql;
pub mod grpc_trpc;
pub mod json;
pub mod middleware;
pub mod modern_frameworks;
pub mod multipart;
pub mod node_compat;
pub mod openapi;
pub mod react;
pub mod responsive_ui;
pub mod security;
pub mod server;
pub mod ssl;
pub mod tailwind;
pub mod template;
pub mod three;
pub mod types;
pub mod vue;
pub mod wasm;
pub mod webgpu;
pub mod websocket;
pub mod pwa;
pub mod workers;

// Realtime
pub mod realtime;

// Code splitting
pub mod code_splitting;

// Bundling
pub mod bundling;

pub use code_splitting::{
    AssetOptimization, CodeSplitAnalyzer, CodeSplitConfig, CodeSplitStrategy,
    ChunkInfo, CommentPreservation, ComponentSplitConfig, CompressionSettings, FontOptimization,
    ImageFormat, ImageOptimization, Minification, MinifyLevel, PrefetchStrategy, RouteConfig,
    RouteSplitConfig, SideEffectAnalysis, SplitAnalysis, SplitError, TreeShaking, TreeShakingStats,
    VendorGroupStrategy, VendorSplitConfig,
    CacheStrategy as CodeSplitCacheStrategy,
};

// Bundling (Vite, Webpack, Esbuild)
pub use bundling::{
    BundleAnalysis, BundleFormat, BundleOptimization, CachingConfig, CacheStrategy as BundleCacheStrategy,
    ChunkInfo as BundleChunkInfo, ChunkType, CodeSplittingConfig, CompressionConfig, DependencyInfo,
    DuplicateModule, EsbuildBuiltinPlugin, EsbuildConfig, EsbuildConfigBuilder, EsbuildLoader,
    EsbuildPlugin, LargeModule, LazyLoadingConfig, MinificationConfig, OptimizationLevel,
    ResolutionStrategy, SourceMapType, SplitPoint, TreeShakingConfig,
    TargetEnvironment, ViteBuiltinPlugin, ViteConfig, ViteConfigBuilder,
    ViteCssConfig, ViteHmrConfig, ViteOptimizeDepsConfig, VitePlugin, VitePluginHooks,
    ViteRollupOutputOptions, ViteRollupOptions, ViteServerConfig, ViteMinifyType, WebpackBuiltinPlugin,
    WebpackChunkIds, WebpackChunksSelection, WebpackConfig, WebpackConfigBuilder, WebpackCacheGroup,
    WebpackDevServerConfig, WebpackEntry, WebpackLoader, WebpackModuleConfig, WebpackModuleIds,
    WebpackOptimizationConfig, WebpackPerformanceConfig, WebpackPlugin, WebpackResolveConfig,
    WebpackRuntimeChunk, WebpackRule, WebpackRuleType, WebpackSplitChunksConfig, WebpackStatsConfig,
    ViteBuildConfig, ViteBundleConfig, ViteAssetsConfig, ViteCorsConfig, ViteHttpsConfig, VitePreviewConfig,
    ViteProxyConfig, ViteResolveConfig, ViteCssModulesConfig, WebpackOutputConfig,
};

// Core types
pub use types::{HeaderMap, Method, Request, Response, StatusCode};

// Server
pub use server::{html_response, json_response, text_response, Handler, Route, Server};

// Client
pub use client::{Client, HttpClient};

// Event loop
pub use event_loop::{run_loop, Event, EventHandler};

// Multipart
pub use multipart::{MultipartData, MultipartField};

// Cookies
pub use cookies::{Cookie, CookieJar, SameSite, Session};

// Forms
pub use forms::{FormData, QueryBuilder, UrlBuilder};

// Middleware
pub use middleware::{Middleware, MiddlewareChain, MiddlewareError, ResponseModifier};

// WASM
pub use wasm::{
    WasmError, WasmHostFunction, WasmImports, WasmInstance, WasmMemory, WasmModule, WasmValue,
    WasmValueType,
};

// WebGPU
pub use webgpu::{
    BufferUsage, ComputePipeline, GpuAdapterInfo, GpuAdapterType, GpuBuffer, GpuDevice,
    GpuDeviceConfig, GpuError, GpuShaderModule, RenderPipeline, TextureFormat, TextureUsage,
    VertexFormat, WgslBuilder,
};

// Full-stack frameworks (Next.js, Remix)
pub use fullstack_frameworks::{
    ApiRoute, ComponentType, HttpMethod, Metadata, NextAppRouter, NextDataFetching, NextLayout,
    NextPage, OpenGraphMetadata, Prop, PropType, RemixAction, RemixLoader, RemixRoute,
    ServerAction, TwitterMetadata,
};

// Vue.js
pub use vue::{
    CompiledTemplate, CompilerOptions, ComputedProperty, Event as VueEvent, LifecycleHook,
    Method as VueMethod, PiniaStore, PropType as VuePropType, ReactiveState, RouterMode,
    TemplateDirective, VueComponent, VueProp, VueRoute, VueRouter, VueTemplate,
    VueTemplateCompiler, WatchSource, Watcher, WatcherOptions,
};

// Responsive UI
pub use responsive_ui::{
    Animation, AnimationSystem, Breakpoint, DarkMode, EasingFunction, GridArea, GridPlacement,
    GridSystem, Keyframe, ResponsiveBreakpoints, ResponsiveStrategy, ResponsiveUISystem,
    ThemeColor, ThemeMode, TrackSize,
};

// gRPC and tRPC
pub use grpc_trpc::{
    GrpcBidiStreamHandler, GrpcClientStreamHandler, TrpcInput, TrpcMutationHandler,
    TrpcProcedure, TrpcProcedureType, TrpcQueryHandler, TrpcResponse, GrpcError, GrpcHandler,
    GrpcMethod, GrpcMethodType, GrpcMetadata, GrpcResponse as GrpcRpcResponse, GrpcServerStreamHandler,
    GrpcService, GrpcServiceBuilder, GrpcStatusCode, GrpcStream, ProtoEnum, ProtoEnumValue, ProtoField,
    ProtoField as ProtoFieldTypeTrait, ProtoFile, ProtoMessage, ProtoMethod, ProtoService,
    ProtoFieldType, TrpcContext, TrpcError as TrpcErrorExport,
    TrpcErrorCode, TrpcRouter, TrpcRouterBuilder, TrpcSubscription, TrpcSubscriptionHandler,
};

// Tailwind CSS
pub use tailwind::{
    BadgeStyles, BorderRadius, ButtonStyles, CardStyles, ColorPalette, ComponentPrimitives,
    DesignSystem, DropdownStyles, FontSizes, InputStyles, LayoutPresets, ModalStyles,
    NavigationStyles, ResponsiveBuilder, Screens, SemanticColors, Shadows, SpacingPresets,
    SpacingScale, TailwindConfig, TailwindUtility, ThemeConfig, TransitionConfig,
    TypographyPresets, ZIndex,
};

// OpenAPI
pub use openapi::{
    ApiGenerator, ApiOperation, Components, Contact, Discriminator, Encoding, Example, ExternalDocumentation,
    Header as OpenApiHeader, HttpMethod as OpenApiMethod, Info, JsonSchemaBuilder, License, MediaType, OAuthFlow, OAuthFlows, OpenAPISpec,
    Operation, Parameter, ParameterLocation, ParameterStyle, PathItem, RequestBody, Response as OpenApiResponse, Schema,
    SchemaBuilder, SecurityScheme, Server as OpenApiServer, ServerVariable, Tag, TargetLanguage, Xml,
};

// GraphQL
pub use graphql::{
    ExecutionContext, Field, FragmentSpread, GraphQLError, GraphQLField, GraphQLInputValue,
    GraphQLQuery, GraphQLResolver, GraphQLResponse, GraphQLResult, GraphQLSchema, GraphQLSubscription,
    GraphQLType, GraphQLValue, InlineFragment, OperationType, Selection, SelectionSet, SubscriptionHandler,
    VariableDefinition,
};

// PWA (Progressive Web App)
pub use pwa::{
    AppShortcut, BackgroundSyncManager, BackgroundSyncRegistration, BackgroundSyncType,
    CacheEntry, CacheStrategy, CacheStrategyType, CacheStorage, CachedResponse, DisplayMode,
    FileHandler, ManifestIcon, ManifestScreenshot, NetworkRequirements, NotificationAction,
    Orientation, ProtocolHandler, PushError, PushManager, PushMessage, PushSubscription,
    PWAPrimitives, RelatedApplication, ServiceWorkerClient, ServiceWorkerRegistration,
    ServiceWorkerState, ShareMethod, ShareParams, ShareTarget, VapidKeys, VisibilityState,
    WebManifest,
};

// Workers (Web Workers and Worklets)
pub use workers::{
    ArgumentDescriptor, AudioWorklet, AudioWorkletInstance, AudioWorkletProcessor,
    PaintClass, PaintOutput, PaintWorklet, ParameterDescriptor, ProcessorState,
    SharedWorker, SharedWorkerMessage, SharedWorkerPort, TaskPriority, WebWorker,
    WorkerError, WorkerMessage, WorkerMessageType, WorkerPool, WorkerPoolStats,
    WorkerState, WorkerTask, WorkerTaskResult,
};

// Realtime (Socket.IO and SSE)
pub use realtime::{
    BroadcastHandler, CallbackHandler, EventData, EventSource, HandshakeInfo,
    KeepAliveHandle, Namespace, Room, SSEEvent, SSEServer, SocketIOClient,
    SocketIOEvent, SocketIOServer, socketio_handshake, sse_response,
    parse_socketio_handshake, SocketPacketType, EnginePacketType,
};

// Re-export EventHandler with alias to avoid conflict with event_loop::EventHandler
pub use realtime::{EventHandler as SocketIOEventHandler, MiddlewareFn as SocketIOMiddlewareFn};

// Modern frameworks (Svelte, SolidJS)
pub use modern_frameworks::{
    DerivedStore, LifecycleHandler, ReactiveStatement, Route as ModernRoute, RouterType, RuneType,
    SolidComponent, SolidContext, SolidEffect, SolidMemo, SolidNode, SolidProp, SolidRouter, SolidSignal,
    StoreType, SvelteComponent, SvelteProp, SvelteState, SvelteStore, TemplateBlock, TemplateNode,
};

// Advanced web (WebAuthn, WebRTC)
pub use advanced_web::{
    AttestationConveyancePreference, AuthenticatorAttachment, AuthenticatorAssertionResponse,
    AuthenticatorAttestationResponse, AuthenticatorData, AuthenticatorSelectionCriteria,
    AuthenticatorTransport, AttestedCredentialData, AuthenticationExtensionsClientInputs,
    AuthenticationExtensionsClientOutputs, BundlePolicy, CredentialAssertionOptions,
    CredentialAssertionResponse, CredentialCreationOptions, CredentialCreationResponse,
    CredentialPropertiesExtension, CredentialPropertiesOutput, CredentialType,
    DataChannelPriority, DataChannelState, IceCandidate, IceCandidateType, IceConnectionState,
    IceGatheringState, IceProtocol, IceServer, IceTransportPolicy, LargeBlobExtension,
    LargeBlobOutput, MediaDeviceInfo, MediaDevices, MediaStream, MediaStreamConstraints,
    MediaStreamTrack, MediaTrackKind, PeerConnectionState, PublicKeyCredentialDescriptor,
    PublicKeyCredentialParameters, RelyingPartyEntity, ResidentKeyRequirement,
    RtcpMuxPolicy, RtcConfiguration, SdpType, SessionDescription, SignalingState,
    StoredCredential, UserEntity, UserVerificationRequirement, WebAuthnConfig, WebAuthnError,
    WebAuthnServer, WebRTCDataChannel, WebRTCPeer,
};

