import "./assets/main.css";

import { createApp } from "vue";
import App from "./App.vue";
import router from "./router";
import { VueQueryPlugin, type VueQueryPluginOptions } from "@tanstack/vue-query";
import PrimeVue from "primevue/config";
import ToastService from "primevue/toastservice";
import Aura from "@primevue/themes/aura";

import { Configuration, DefaultConfig, type Middleware, type RequestContext, type ResponseContext } from "@/api";
import { getCookie, useBackendHost } from "@/utils";
// Import for side-effect: opens the WebSocket and starts populating the
// reactive event-stream state used by status indicators.
import "@/composables/useEventStream";
/**
 * Small middleware to add appropriate headers depending on the HTTP method
 * before starting the API call.
 */
export class AppropriateOptionsMiddleware implements Middleware {
    /**
     * Called before executing the request.
     * @param context
     */
    pre(context: RequestContext) {
        const init = context.init;

        const currentHeaders =
            Array.isArray(init.headers) || init.headers instanceof Headers
                ? Object.fromEntries(init.headers)
                : init.headers;

        context.init = {
            ...init,
            headers: {
                ...currentHeaders,
                "X-CSRFToken": getCookie("csrftoken") ?? "",
                "Content-Type": "application/json",
            },
        };
        return Promise.resolve({ url: context.url, init: context.init });
    }
}

const app = createApp(App);

const backendHost = useBackendHost();
DefaultConfig.config = new Configuration({
    basePath: backendHost,
    credentials: "include",
    middleware: [new AppropriateOptionsMiddleware()],
});

app.use(router);
app.use(PrimeVue, { theme: { preset: Aura } });
app.use(ToastService);
const vueQueryPluginOptions: VueQueryPluginOptions = {
    queryClientConfig: {
        defaultOptions: {
            queries: {
                throwOnError: true,
                refetchOnWindowFocus: false,
                retry: !import.meta.env.VITE_CI, // Disable retries during tests
            },
        },
    },
};
app.use(VueQueryPlugin, vueQueryPluginOptions);

app.mount("#app");
