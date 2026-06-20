//! Vue.js Synthesis Demo
//!
//! Demonstrates Vue 3 component generation with Composition API, Vue Router, and Pinia stores.

use nsynth::http::vue::{
    LifecycleHook, PiniaStore, PropType, ReactiveState, RouterMode, TemplateDirective,
    VueComponent, VueRoute, VueRouter, VueTemplate, WatchSource, WatcherOptions,
};

fn main() {
    println!("=== Vue.js Synthesis Demo ===\n");

    // Example 1: Simple Counter Component
    println!("1. Counter Component:");
    let counter = VueComponent::new("Counter")
        .add_ref("count", PropType::Number, serde_json::json!(0))
        .computed("doubled", PropType::Number, "count.value * 2")
        .method("increment", vec![], "count.value++")
        .method("decrement", vec![], "count.value--")
        .emit("change", Some(PropType::Number));

    println!("{}\n", counter.to_vue3());

    // Example 2: User Form Component
    println!("2. User Form Component:");
    let mut props = std::collections::HashMap::new();
    props.insert("username".to_string(), PropType::String);
    props.insert("email".to_string(), PropType::String);

    let mut initial = serde_json::Map::new();
    initial.insert("username".to_string(), serde_json::json!(""));
    initial.insert("email".to_string(), serde_json::json!(""));

    let user_form = VueComponent::new("UserForm")
        .prop("submitUrl", PropType::String)
        .reactive("form", props, serde_json::json!(initial))
        .watch(
            WatchSource::Ref("form.username".to_string()),
            "console.log('Username changed:', newVal)",
        )
        .action(
            "submit",
            vec![("data".to_string(), PropType::Object)],
            "fetch(props.submitUrl, { method: 'POST', body: JSON.stringify(data) })",
        );

    println!("{}\n", user_form.to_vue3());

    // Example 3: Todo List Component
    println!("3. Todo List Component:");
    let todo_list = VueComponent::new("TodoList")
        .add_ref("todos", PropType::Array, serde_json::json!([]))
        .add_ref("newTodo", PropType::String, serde_json::json!(""))
        .computed("todoCount", PropType::Number, "todos.value.length")
        .computed("hasTodos", PropType::Boolean, "todoCount.value > 0")
        .method("addTodo", vec![],
                "if (newTodo.value.trim()) { todos.value.push({ id: Date.now(), text: newTodo.value, done: false }); newTodo.value = ''; }")
        .method("removeTodo", vec![("id".to_string(), PropType::Number)],
                "todos.value = todos.value.filter(t => t.id !== id)")
        .lifecycle(LifecycleHook::Mounted("console.log('TodoList mounted')".to_string()))
        .template(VueTemplate {
            content: "<ul v-for=\"todo in todos\" :key=\"todo.id\">\n  <li>{{ todo.text }}</li>\n</ul>".to_string(),
            directives: vec![
                TemplateDirective::For { item: "todo".to_string(), source: "todos".to_string() },
            ],
        });

    println!("{}\n", todo_list.to_vue3());

    // Example 4: Vue Router Configuration
    println!("4. Vue Router Configuration:");
    let router = VueRouter::new()
        .route(
            VueRoute::new("/", "HomeView")
                .with_name("home")
                .meta("title", serde_json::json!("Home")),
        )
        .route(
            VueRoute::new("/users/:id", "UserView")
                .with_name("user")
                .with_props()
                .meta("requiresAuth", serde_json::json!(true)),
        )
        .route(
            VueRoute::new("/admin", "AdminLayout")
                .child(VueRoute::new("dashboard", "AdminDashboard"))
                .child(VueRoute::new("users", "AdminUsers")),
        )
        .with_mode(RouterMode::History);

    println!("{}\n", router.to_router());

    // Example 5: Pinia Store
    println!("5. Pinia Store:");
    let store = PiniaStore::new("cart")
        .state("items", PropType::Array, serde_json::json!([]))
        .state("total", PropType::Number, serde_json::json!(0))
        .getter("itemCount", PropType::Number, "items.length")
        .getter("isEmpty", PropType::Boolean, "itemCount === 0")
        .action(
            "addItem",
            vec![("item".to_string(), PropType::Object)],
            "items.push({ ...item, id: Date.now() }); updateTotal();",
        )
        .action(
            "removeItem",
            vec![("id".to_string(), PropType::Number)],
            "items = items.filter(i => i.id !== id); updateTotal();",
        )
        .action(
            "updateTotal",
            vec![],
            "total = items.reduce((sum, item) => sum + item.price, 0);",
        )
        .action("clear", vec![], "items = []; total = 0;");

    println!("{}\n", store.to_store());

    println!("=== Demo Complete ===");
}

// Helper function for the example
trait VueComponentExt {
    fn action(&self, name: &str, params: Vec<(String, PropType)>, body: &str) -> VueComponent;
}

impl VueComponentExt for VueComponent {
    fn action(&self, name: &str, params: Vec<(String, PropType)>, body: &str) -> VueComponent {
        self.clone().method(name, params, body)
    }
}
