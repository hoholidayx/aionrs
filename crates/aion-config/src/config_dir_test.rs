use super::*;

#[test]
fn defaults_to_hidden_directory_on_fresh_install() {
    let home = tempfile::tempdir().unwrap();
    let preferred = home.path().join(".aionrs");
    let legacy = home.path().join("legacy");
    assert_eq!(resolve_config_dir(home.path(), Some(legacy)), preferred);
}

#[test]
fn moves_existing_config_and_memory_once() {
    let home = tempfile::tempdir().unwrap();
    let preferred = home.path().join(".aionrs");
    let legacy = home.path().join("legacy");
    let memory = "projects/project/memory/MEMORY.md";
    fs::create_dir_all(legacy.join(memory).parent().unwrap()).unwrap();
    fs::write(legacy.join(memory), "saved memory").unwrap();
    fs::write(legacy.join("AGENTS.md"), "saved instructions").unwrap();

    assert_eq!(resolve_config_dir(home.path(), Some(legacy.clone())), preferred);
    assert_eq!(resolve_config_dir(home.path(), Some(legacy.clone())), preferred);
    assert_eq!(fs::read_to_string(preferred.join(memory)).unwrap(), "saved memory");
    assert_eq!(
        fs::read_to_string(preferred.join("AGENTS.md")).unwrap(),
        "saved instructions"
    );
    assert!(!legacy.exists());
}

#[test]
fn leaves_both_directories_intact_if_the_new_root_exists() {
    let home = tempfile::tempdir().unwrap();
    let preferred = home.path().join(".aionrs");
    let legacy = home.path().join("legacy");
    fs::create_dir(&preferred).unwrap();
    fs::create_dir(&legacy).unwrap();
    fs::write(preferred.join("config.toml"), "new").unwrap();
    fs::write(legacy.join("config.toml"), "old").unwrap();

    assert_eq!(resolve_config_dir(home.path(), Some(legacy.clone())), preferred);
    assert_eq!(fs::read_to_string(preferred.join("config.toml")).unwrap(), "new");
    assert_eq!(fs::read_to_string(legacy.join("config.toml")).unwrap(), "old");
}

#[test]
fn preserves_a_legacy_relative_symlink() {
    let home = tempfile::tempdir().unwrap();
    let preferred = home.path().join(".aionrs");
    let legacy_parent = home.path().join("old");
    let legacy = legacy_parent.join("aionrs");
    let target = legacy_parent.join("target");
    fs::create_dir_all(&target).unwrap();
    std::os::unix::fs::symlink("target", &legacy).unwrap();

    assert_eq!(resolve_config_dir(home.path(), Some(legacy.clone())), preferred);
    assert_eq!(fs::canonicalize(preferred).unwrap(), fs::canonicalize(legacy).unwrap());
}
